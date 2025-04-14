import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_lib.agents.SAC.network import Network_graph

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], device='cpu'):
        super(QNetwork, self).__init__()
        self.device = device
        self.q_net = Network_graph(state_dim + action_dim, 1, hidden_dims, activation_fn=nn.ReLU(), device=device)
        self.to(device)

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=-1)
        return self.q_net(x)