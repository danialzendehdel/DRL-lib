import torch
import torch.nn as nn
import torch.nn.functional as F

class Network_graph(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn=F.relu, device='cpu'):
        super(Network_graph, self).__init__()

        self.device = device
        self.input_layer = nn.Linear(input_dim, hidden_dims[0]).to(device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]).to(device) for i in range(len(hidden_dims) - 1)])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim).to(device)
        self.activation_fn = activation_fn

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.input_layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension for single inputs

        # Apply input layer and activation
        x = self.input_layer(x)
        x = self.activation_fn(x)
        
        # Apply hidden layers with activation
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(x)
            
        # Apply output layer (no activation)
        x = self.output_layer(x)
        return x