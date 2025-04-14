import torch
import torch.nn as nn
import torch.optim as optim
from drl_lib.utils.replay_buffer import ReplayBuffer
from drl_lib.agents.SAC.policy_nn import Actor
from drl_lib.agents.SAC.value_network import ValueNetwork
from drl_lib.agents.SAC.Q_network import QNetwork
import torch.nn.functional as F
from drl_lib.debugging.journaling import Journal


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dims, alpha, gamma, tau, actor_lr, critic_lr, value_lr, buffer_size, batch_size, device, action_bound=None):
        """
        Initialize the SAC agent with all necessary components.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dims (list): List of hidden layer sizes for the networks.
            alpha (float): Temperature parameter for entropy regularization.
            gamma (float): Discount factor.
            tau (float): Target network update rate.
            actor_lr (float): Learning rate for the Actor.
            critic_lr (float): Learning rate for the Q-Networks.
            value_lr (float): Learning rate for the Value Network.
            buffer_size (int): Maximum size of the Replay Buffer.
            batch_size (int): Batch size for sampling from the Replay Buffer.
            device (str): Device to run computations on (e.g., 'cuda' or 'cpu').
            action_bound (tuple, optional): (min, max) bounds for actions (e.g., (-2, 2)).
        """
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        
        # Initialize Journal for monitoring
        self.journal = Journal(
            directory="logs",
            action_bounds=action_bound if action_bound else (-1, 1),
            experiment_name="sac_experiment"
        )
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dims, device=device, action_bound=action_bound)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.value = ValueNetwork(state_dim, hidden_dims, device=device)
        self.target_value = ValueNetwork(state_dim, hidden_dims, device=device)
        self.target_value.load_state_dict(self.value.state_dict())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, batch_size, device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state (np.ndarray): Current state.
            deterministic (bool): Whether to select the mean action (default: False).
        
        Returns:
            np.ndarray: Selected action.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(state, deterministic=deterministic)
        action = action.squeeze(0).detach().cpu().numpy()
        
        # Check action bounds
        self.journal._actor_debug(action)
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool or int): Done flag.
        """
        self.replay_buffer.store(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update the networks using a batch from the replay buffer.
        """
        # Check if we have enough samples for a batch
        if self.replay_buffer.size < self.replay_buffer.batch_size:
            self.journal.log(f"Not enough samples in replay buffer for update. Have {self.replay_buffer.size}, need {self.replay_buffer.batch_size}", "WARNING")
            return  # Not enough samples to update
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Update Q-networks
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.q1(next_states, next_actions)
            q2_next = self.q2(next_states, next_actions)
            q_next_min = torch.min(q1_next, q2_next)
            v_target_next = q_next_min - self.alpha * next_log_probs
            y = rewards + self.gamma * (1 - dones) * v_target_next
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)
        
        # Check gradients before update
        self.journal._check_gradient(self.q1)
        self.journal._check_gradient(self.q2)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update Value network
        actions_sampled, log_probs = self.actor.sample(states)
        q1_sampled = self.q1(states, actions_sampled)
        q2_sampled = self.q2(states, actions_sampled)
        q_min_sampled = torch.min(q1_sampled, q2_sampled)
        v_target = (q_min_sampled - self.alpha * log_probs).detach()
        
        v_pred = self.value(states)
        v_loss = F.mse_loss(v_pred, v_target)
        
        # Check gradients before update
        self.journal._check_gradient(self.value)
        
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()
        
        # Update Actor
        actions_reparam, log_probs_reparam = self.actor.sample(states)
        q1_reparam = self.q1(states, actions_reparam)
        q2_reparam = self.q2(states, actions_reparam)
        q_min_reparam = torch.min(q1_reparam, q2_reparam)
        policy_loss = (self.alpha * log_probs_reparam - q_min_reparam).mean()
        
        # Check gradients before update
        self.journal._check_gradient(self.actor)
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update target Value network
        self._update_target_network()
        
        # Log update information
        self.journal.log(f"Update completed - Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, V Loss: {v_loss.item():.4f}, Policy Loss: {policy_loss.item():.4f}")
        
    def _update_target_network(self):
        """
        Soft update the target Value network.
        """
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)