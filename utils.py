import torch


def load_adj(filepath):
    """
    Load an adjacency matrix from a .pt file.

    Args:
        filepath (str): Path to the adjacency matrix file.

    Returns:
        torch.Tensor: Dense adjacency matrix as a PyTorch tensor.
    """
    # Load tensor directly from .pt file
    adj = torch.load(filepath)
    if not isinstance(adj, torch.Tensor):
        raise ValueError(f"File {filepath} does not contain a valid PyTorch tensor.")
    return adj


def sparse_to_dense(sparse_adj):
    """
    Convert a sparse adjacency matrix to a dense format.

    Args:
        sparse_adj: PyTorch sparse tensor.

    Returns:
        torch.Tensor: Dense PyTorch tensor.
    """
    return sparse_adj.to_dense()


def preprocess_adj(adj):
    """
    Normalize an adjacency matrix.

    Args:
        adj: Dense adjacency matrix.

    Returns:
        torch.Tensor: Normalized adjacency matrix.
    """
    adj = adj + torch.eye(adj.shape[0])  # Add self-loops
    row_sum = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
