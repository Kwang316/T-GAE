import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) Convolution Layer.
    """
    def __init__(self, input_dim, output_dim):
        super(GINConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, A, X):
        """
        Forward pass for GINConv.
        
        Args:
            A (torch.Tensor): Adjacency matrix.
            X (torch.Tensor): Node features.
        
        Returns:
            torch.Tensor: Updated node features.
        """
        X = self.linear(X + A @ X)
        X = F.relu(X)
        return X


class TGAE_Encoder(nn.Module):
    """
    Encoder for Transferable Graph Autoencoder (TGAE).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        """
        Initialize the TGAE Encoder.
        
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (list or int): Dimensions of hidden layers. If int, repeated for all layers.
            output_dim (int): Dimension of output features.
            num_hidden_layers (int): Number of hidden layers.
        """
        super(TGAE_Encoder, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_hidden_layers  # Repeat the same size for all layers
        
        self.in_proj = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim[i], hidden_dim[i + 1]) for i in range(len(hidden_dim) - 1)
        ])
        self.out_proj = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, A, X):
        """
        Forward pass for TGAE Encoder.
        
        Args:
            A (torch.Tensor): Adjacency matrix.
            X (torch.Tensor): Node features.
        
        Returns:
            torch.Tensor: Encoded node features.
        """
        X = self.in_proj(X)
        for layer in self.hidden_layers:
            X = F.relu(layer(X))
        X = self.out_proj(X)
        return X


class TGAE(nn.Module):
    """
    Transferable Graph Autoencoder (TGAE).
    """
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        """
        Initialize the TGAE model.
        
        Args:
            num_hidden_layers (int): Number of hidden layers in the encoder.
            input_dim (int): Dimension of input features.
            hidden_dim (int or list): Dimensions of hidden layers.
            output_dim (int): Dimension of output features.
        """
        super(TGAE, self).__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers)

    def forward(self, X, adj):
        """
        Forward pass for TGAE.
        
        Args:
            X (torch.Tensor): Node features.
            adj (torch.Tensor): Adjacency matrix.
        
        Returns:
            torch.Tensor: Encoded features.
        """
        Z = self.encoder(adj, X)
        return Z
