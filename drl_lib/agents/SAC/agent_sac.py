import torch
import torch.nn as nn
import torch.optim as optim
from drl_lib.utils.replay_buffer import ReplayBuffer
from drl_lib.agents.SAC.policy_nn import Actor
from drl_lib.agents.SAC.Q_network import QNetwork
import torch.nn.functional as F
from drl_lib.debugging.journaling import Journal


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dims, alpha, gamma, tau, actor_lr, critic_lr, buffer_size, batch_size, device, action_bound=None, use_tanh=True, value_lr=None):
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
            buffer_size (int): Maximum size of the Replay Buffer.
            batch_size (int): Batch size for sampling from the Replay Buffer.
            device (str): Device to run computations on (e.g., 'cuda' or 'cpu').
            action_bound (tuple, optional): (min, max) bounds for actions (e.g., (-2, 2)). Defaults to None.
            use_tanh (bool, optional): Whether to use tanh squashing for actions in the Actor. Defaults to True.
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_tanh = use_tanh

        # Initialize Journal for monitoring
        self.journal = Journal(
            directory="logs",
            action_bounds=action_bound if action_bound else (-1, 1),
            experiment_name="sac_experiment"
        )
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dims, device=device, action_bound=action_bound, use_tanh=self.use_tanh)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dims, device=device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, batch_size, device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        
        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)
        self.alpha = self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(state, deterministic=deterministic)
        action = action.squeeze(0).detach().cpu().numpy()
        self.journal._actor_debug(action)
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)
    
    def update(self):
        if self.replay_buffer.size < self.replay_buffer.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next_target = self.q1_target(next_states, next_actions)
            q2_next_target = self.q2_target(next_states, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_next_target

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor and Alpha update
        for params in self.q1.parameters():
            params.requires_grad = False
        for params in self.q2.parameters():
            params.requires_grad = False

        actions_pi, log_probs = self.actor.sample(states)
        q1_pi = self.q1(states, actions_pi)
        q2_pi = self.q2(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        for params in self.q1.parameters():
            params.requires_grad = True
        for params in self.q2.parameters():
            params.requires_grad = True
        
        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self):
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)