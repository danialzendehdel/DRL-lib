import torch
import torch.nn as nn
import torch.nn.functional as F
from drl_lib.agents.SAC.network import Network_graph

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256], device='cpu'):
        super(ValueNetwork, self).__init__()
        self.device = device
        self.value_net = Network_graph(state_dim, 1, hidden_dims, activation_fn=F.relu, device=device)
        self.to(device)  # Ensure the network is on the correct device

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension for single states

        return self.value_net(state)