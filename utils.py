import torch
from scipy.sparse import coo_matrix
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
    print("Preprocessing graph...")
    # Convert adjacency matrix to COO format if necessary
    if not isinstance(adj, coo_matrix):
        adj = coo_matrix(adj.cpu().numpy())
    
    # Add self-loops
    adj_ = adj + coo_matrix(np.eye(adj.shape[0]))
    
    # Compute the degree matrix
    rowsum = np.array(adj_.sum(1)).flatten()
    degree_mat_inv_sqrt = np.power(rowsum, -0.5)
    degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.0
    degree_mat_inv_sqrt = coo_matrix(np.diag(degree_mat_inv_sqrt))
    
    # Normalize the adjacency matrix
    adj_normalized = degree_mat_inv_sqrt @ adj_ @ degree_mat_inv_sqrt

    # Convert to PyTorch sparse tensor
    adj_normalized = coo_matrix(adj_normalized)
    indices = torch.tensor(np.vstack((adj_normalized.row, adj_normalized.col)), dtype=torch.long)
    values = torch.tensor(adj_normalized.data, dtype=torch.float)
    shape = adj_normalized.shape
    
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape))


def save_mapping(mapping, output_file):
    """
    Save the computed node mapping to a file.
    """
    with open(output_file, "w") as f:
        for i, j in tqdm(enumerate(mapping), desc="Saving node mapping", total=len(mapping)):
            f.write(f"{i} -> {j.item()}\n")
    print(f"Node mapping saved to {output_file}")
