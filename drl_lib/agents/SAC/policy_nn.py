import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_lib.agents.SAC.network import Network_graph

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, activation_fn=nn.ReLU(), device='cpu', action_bound=None, use_tanh=True):
        """
        Actor (Policy) Network for SAC.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dims (list): List of hidden layer sizes.
            activation_fn (torch.nn.functional, optional): Activation function. Defaults to nn.ReLU().
            device (str, optional): Device to run computations on. Defaults to 'cpu'.
            action_bound (tuple, optional): (min, max) bounds for actions. Defaults to None.
            use_tanh (bool, optional): Whether to use tanh squashing for actions. Defaults to True.
        """
        super(Actor, self).__init__()

        self.device = device
        if action_bound is not None:
            self.action_bound = (
                torch.tensor(action_bound[0], dtype=torch.float32, device=device),
                torch.tensor(action_bound[1], dtype=torch.float32, device=device)
            )
        else:
            self.action_bound = None
        self.action_dim = action_dim
        self.use_tanh = use_tanh
        self.min_logstd, self.max_logstd = -20, 2

        # Initialize the network with correct output size
        self.policy_net = Network_graph(state_dim, action_dim * 2, hidden_dims, activation_fn, device)
        self._init_weights()  # Call weight initialization

    def _init_weights(self):
        """Initialize network weights for better training stability."""
        for module in self.policy_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Assuming Network_graph has an output_layer; adjust if incorrect
        try:
            nn.init.uniform_(self.policy_net.output_layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(self.policy_net.output_layer.bias)
        except AttributeError:
            print("Warning: Network_graph has no output_layer; adjust initialization.")

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        output = self.policy_net(state)
        mu, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=self.min_logstd, max=self.max_logstd)
        return mu, log_std
    
    def sample(self, state, deterministic=False):
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Input state (tensor or array).
            deterministic (bool): If True, return the mean action without sampling.
        
        Returns:
            action (tensor): Sampled or deterministic action, scaled to action bounds.
            log_prob (tensor or None): Log probability of the action (None if deterministic).
        """
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if deterministic:
            # If deterministic, use mean. Apply tanh if use_tanh is True.
            u = mu
            if self.use_tanh:
                action = torch.tanh(u)
            else:
                action = u
            log_prob = None  # No log_prob for deterministic actions
        else:
            # If not deterministic, sample from the distribution.
            u = dist.rsample()  # Reparameterized sampling
            if self.use_tanh:
                action = torch.tanh(u)
                # Log probability correction for tanh squashing
                log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            else:
                action = u
                # No squashing, log_prob is just from the Normal distribution
                log_prob = dist.log_prob(u)

            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale action to environment bounds if provided
        if self.action_bound is not None:
            low, high = self.action_bound
            if self.use_tanh: # If tanh is used, actions are in (-1, 1) before scaling
                action_scaled = low + (high - low) * (action + 1) / 2
            else: # If tanh is not used, actions are unbounded from the Gaussian
                  # We directly clip them if action_bound is specified.
                  # Note: If action_bound is None and use_tanh is False, actions are truly unbounded.
                action_scaled = action

            action = torch.clamp(action_scaled, min=low, max=high)
        elif not self.use_tanh:
            # If no tanh and no action_bound, action is u directly
            action = u


        return action, log_prob
    



if __name__ == "__main__":
    # Example with tanh (default)
    actor_tanh = Actor(state_dim=4, action_dim=2, hidden_dims=[256, 256], action_bound=(-2, 2), device='cpu', use_tanh=True)
    state = np.random.randn(4)
    mu_tanh, log_std_tanh = actor_tanh.forward(state)
    action_tanh, log_prob_tanh = actor_tanh.sample(state)
    print("With Tanh:")
    print("mu:", mu_tanh)
    print("log_std:", log_std_tanh)
    print("action:", action_tanh)
    print("log_prob:", log_prob_tanh)
    print("-" * 30)

    # Example without tanh and with action bounds
    actor_no_tanh_bounded = Actor(state_dim=4, action_dim=2, hidden_dims=[256, 256], action_bound=(-3, 3), device='cpu', use_tanh=False)
    mu_no_tanh_b, log_std_no_tanh_b = actor_no_tanh_bounded.forward(state)
    action_no_tanh_b, log_prob_no_tanh_b = actor_no_tanh_bounded.sample(state)
    print("Without Tanh (Bounded):")
    print("mu:", mu_no_tanh_b)
    print("log_std:", log_std_no_tanh_b)
    print("action:", action_no_tanh_b)
    print("log_prob:", log_prob_no_tanh_b)
    print("-" * 30)

    # Example without tanh and without action bounds
    actor_no_tanh_unbounded = Actor(state_dim=4, action_dim=2, hidden_dims=[256, 256], action_bound=None, device='cpu', use_tanh=False)
    mu_no_tanh_u, log_std_no_tanh_u = actor_no_tanh_unbounded.forward(state)
    action_no_tanh_u, log_prob_no_tanh_u = actor_no_tanh_unbounded.sample(state)
    print("Without Tanh (Unbounded):")
    print("mu:", mu_no_tanh_u)
    print("log_std:", log_std_no_tanh_u)
    print("action:", action_no_tanh_u)
    print("log_prob:", log_prob_no_tanh_u)

    actor = Actor(state_dim=4, action_dim=2, hidden_dims=[256, 256], action_bound=(-1, 1), device='cpu')
    state = np.random.randn(4)
    mu, log_std = actor.forward(state)
    action, log_prob = actor.sample(state)
    print("mu:", mu)
    print("log_std:", log_std)
    print("action:", action)
    print("log_prob:", log_prob)