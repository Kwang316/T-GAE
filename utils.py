import torch
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from tqdm import tqdm

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

def preprocess_graph(adj):
    """
    Preprocess adjacency matrix for use in TGAE.
    """
    adj = coo_matrix(adj.cpu().numpy())
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.tensor(adj_normalized, dtype=torch.float)

def save_mapping(mapping, output_file):
    """
    Save the computed node mapping to a file.
    """
    with open(output_file, "w") as f:
        for i, j in enumerate(mapping):
            f.write(f"{i} -> {j.item()}\n")
    print(f"Node mapping saved to {output_file}")
