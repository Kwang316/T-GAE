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
    adj = coo_matrix(adj.cpu().numpy())
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    # Add tqdm for preprocessing steps
    with tqdm(total=3, desc="Preprocessing steps") as pbar:
        adj_normalized = torch.tensor(adj_normalized, dtype=torch.float)
        pbar.update(1)  # Step 1 completed
        adj_normalized = adj_normalized.coalesce()  # Step 2: Ensure sparse format
        pbar.update(1)  # Step 2 completed
        # Return preprocessed tensor
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
