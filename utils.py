import torch
from scipy.sparse import coo_matrix
import numpy as np

def load_adj(dataset):
    """
    Load the adjacency matrix from the dataset.
    """
    if dataset.endswith('.pt'):
        print(f"Loading adjacency matrix from {dataset}...")
        data = torch.load(dataset, weights_only=False)  # Explicitly set weights_only
        if 'adj_matrix' in data:
            return data['adj_matrix'], None
        else:
            raise ValueError("The .pt file must contain an 'adj_matrix' key.")
    elif dataset == "celegans":
        S = torch.load("data/celegans.pt")
    elif dataset == "arenas":
        S = torch.load("data/arenas.pt")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def preprocess_graph(adj):
    """
    Normalize the adjacency matrix.
    """
    adj = coo_matrix(adj)
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.tensor(adj_normalized, dtype=torch.float)
