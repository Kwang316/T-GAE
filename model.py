import torch
import torch.nn as nn


class GINConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adj, features):
        return nn.ReLU()(self.linear(features + torch.matmul(adj, features)))


class TGAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([GINConv(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):
            self.layers.append(GINConv(hidden_dims[i - 1], hidden_dims[i]))
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, adj, features):
        for layer in self.layers:
            features = layer(adj, features)
        return self.out_layer(features)


class TGAE(nn.Module):
    def __init__(self, hidden_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim)

    def forward(self, features, adj):
        return self.encoder(adj, features)
