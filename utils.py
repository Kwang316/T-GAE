import torch
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from tqdm import tqdm

import numpy as np
import torch

def load_features(filepath):
    """
    Load node features from a file.

    Args:
        filepath (str): Path to the file containing node features.

    Returns:
        torch.Tensor: Tensor containing node features.
    """
    # Example: If features are stored in a numpy file
    features = np.load(filepath)  # Adjust based on file format
    return torch.tensor(features, dtype=torch.float32)


def load_adj(filepath):
    """
    Load adjacency matrix from a .pt file.
    """
    print(f"Loading adjacency matrix from {filepath}...")
    data = torch.load(filepath)
    if not isinstance(data, dict) or 'adj_matrix' not in data:
        raise ValueError(f"Unsupported file format or missing 'adj_matrix' key in {filepath}.")
    adj_matrix = data['adj_matrix']
    return adj_matrix, data.get('node_mapping', None)

import torch
import numpy as np
from scipy.sparse import coo_matrix

def preprocess_graph(adj):
    """
    Preprocess the adjacency matrix for TGAE.
    """
    print("Preprocessing graph...")
    
    # Convert to COO format if necessary
    if not isinstance(adj, coo_matrix):
        adj = coo_matrix(adj)

    # Add self-loops
    adj_ = adj + coo_matrix(np.eye(adj.shape[0]))

    # Normalize the adjacency matrix
    rowsum = np.array(adj_.sum(1)).flatten()
    degree_mat_inv_sqrt = np.power(rowsum, -0.5)
    degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.  # Avoid inf
    degree_mat_inv_sqrt = np.diag(degree_mat_inv_sqrt)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    # Check if sparse or dense is needed
    if torch.cuda.is_available():
        # Directly use dense tensor on CUDA for speed
        adj_normalized_tensor = torch.tensor(adj_normalized.toarray(), dtype=torch.float32)
    else:
        # Convert to sparse COO tensor for CPU
        adj_normalized = coo_matrix(adj_normalized)
        indices = torch.tensor([adj_normalized.row, adj_normalized.col], dtype=torch.long)
        values = torch.tensor(adj_normalized.data, dtype=torch.float32)
        shape = torch.Size(adj_normalized.shape)
        adj_normalized_tensor = torch.sparse_coo_tensor(indices, values, shape)

    return adj_normalized_tensor


def save_mapping(mapping, output_file):
    """
    Save the computed node mapping to a file.
    """
    with open(output_file, "w") as f:
        for i, j in enumerate(mapping):
            f.write(f"{i} -> {j.item()}\n")
    print(f"Node mapping saved to {output_file}")
