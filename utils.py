import numpy as np
import torch
from scipy.sparse import load_npz, csr_matrix


def load_adj(filepath):
    """
    Load an adjacency matrix from a file.

    Args:
        filepath (str): Path to the adjacency matrix file.

    Returns:
        torch.Tensor: Dense adjacency matrix as a PyTorch tensor.
    """
    if filepath.endswith('.npz'):
        # Load sparse adjacency matrix
        adj_sparse = load_npz(filepath)
        adj_dense = adj_sparse.toarray()  # Convert to dense format
        return torch.tensor(adj_dense, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Expected .npz.")


def sparse_to_dense(sparse_adj):
    """
    Convert a sparse adjacency matrix to a dense format.

    Args:
        sparse_adj: Scipy sparse matrix.

    Returns:
        torch.Tensor: Dense PyTorch tensor.
    """
    dense_adj = sparse_adj.toarray()
    return torch.tensor(dense_adj, dtype=torch.float32)


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
