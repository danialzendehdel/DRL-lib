import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_lib.agents.SAC.network import Network_graph

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, activation_fn=nn.ReLU(), device='cpu', action_bound=None):
        super(Actor, self).__init__()

        self.device = device 
        self.action_bound = action_bound
        self.action_dim = action_dim
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

        if deterministic:
            action = torch.tanh(mu)
            log_prob = None
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()  # Reparameterized sampling
            action = torch.tanh(u)
            log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale action to environment bounds if provided
        if self.action_bound is not None:
            low, high = self.action_bound
            action = low + (high - low) * (action + 1) / 2
            action = torch.clamp(action, min=low, max=high)

        return action, log_prob
    



if __name__ == "__main__":
    actor = Actor(state_dim=4, action_dim=2, hidden_dims=[256, 256], graph=None, action_bound=(-1, 1), device='cpu')
    state = np.random.randn(4)
    mu, log_std = actor.forward(state)
    action, log_prob = actor.sample(state)
    print("mu:", mu)
    print("log_std:", log_std)
    print("action:", action)
    print("log_prob:", log_prob)