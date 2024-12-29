import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, A, X):
        X = self.linear(X + A @ X)
        return F.relu(X)


class TGAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super().__init__()
        if len(hidden_dim) < num_hidden_layers - 2:
            raise ValueError(f"`hidden_dim` must have at least {num_hidden_layers - 2} elements.")

        self.in_proj = nn.Linear(input_dim, hidden_dim[0])
        self.convs = nn.ModuleList(
            GINConv(input_dim + hidden_dim[i], hidden_dim[i + 1]) for i in range(num_hidden_layers - 2)
        )
        self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

    def forward(self, A, X):
        X = self.in_proj(X)
        hidden_states = [X]
        for conv in self.convs:
            X = conv(A, torch.cat([hidden_states[0], X], dim=1))
            hidden_states.append(X)
        X = torch.cat(hidden_states, dim=1)
        return self.out_proj(X)


class TGAE(nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers)

    def forward(self, X, adj):
        return self.encoder(adj, X)
