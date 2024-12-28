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

    # Convert to SciPy sparse matrix (if not already)
    adj = coo_matrix(adj.cpu().numpy())

    # Add self-loops
    adj_ = adj + coo_matrix(np.eye(adj.shape[0]))
    rowsum = np.array(adj_.sum(1)).flatten()

    # Calculate D^(-1/2) (degree matrix inverse square root)
    degree_mat_inv_sqrt = coo_matrix(np.diag(np.power(rowsum, -0.5)))

    # Normalize adjacency matrix: A_norm = D^(-1/2) * A * D^(-1/2)
    with tqdm(total=3, desc="Preprocessing steps") as pbar:
        adj_normalized = degree_mat_inv_sqrt @ adj_ @ degree_mat_inv_sqrt  # Step 1: Normalize
        pbar.update(1)

        # Convert normalized adjacency matrix to sparse tensor
        adj_normalized = torch.sparse_coo_tensor(
            np.vstack((adj_normalized.row, adj_normalized.col)),
            adj_normalized.data,
            torch.Size(adj_normalized.shape),
            dtype=torch.float
        )
        pbar.update(1)  # Step 2 completed

        # Coalesce (ensure well-formed sparse tensor)
        adj_normalized = adj_normalized.coalesce()
        pbar.update(1)  # Step 3 completed

    return adj_normalized

def save_mapping(mapping, output_file):
    """
    Save the computed node mapping to a file.
    """
    with open(output_file, "w") as f:
        for i, j in tqdm(enumerate(mapping), desc="Saving node mapping", total=len(mapping)):
            f.write(f"{i} -> {j.item()}\n")
    print(f"Node mapping saved to {output_file}")
