import torch
import torch.nn as nn

class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, A, X):
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)
        return X

class TGAE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINConv(input_dim, hidden_dim[0]))
        for i in range(1, n_layers - 1):
            self.layers.append(GINConv(hidden_dim[i - 1], hidden_dim[i]))
        self.out_proj = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, A, X):
        for layer in self.layers:
            X = layer(A, X)
        return self.out_proj(X)

class TGAE(torch.nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers)

    def forward(self, X, adj):
        return self.encoder(adj, X)
