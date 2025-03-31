import torch 
import numpy 
import torch.nn as nn 
import torch.nn.functional as F


# General Network 
class Network_graph(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, activation_fn=nn.ReLU(), device='cpu'):
        super(Network_graph, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0]).to(device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]).to(device) for i in range(len(hidden_dims) - 1)])
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim).to(device)

        self.activation_fn = activation_fn
        self.device = device 

    def forward(self, x):
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.squeeze(0)

        x = self.activation_fn(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))

        x = self.output_layer(x)  # Logits which are not normalized 

        return x 
        
        

