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
    Normalize the adjacency matrix.
    """
    # Convert PyTorch tensor to numpy if needed
    if torch.is_tensor(adj):
        adj = adj.cpu().numpy()
    elif isinstance(adj, (coo_matrix, csr_matrix)):
        adj = adj.toarray()
    elif not isinstance(adj, np.ndarray):
        adj = np.array(adj)
        
    # Add self-loops
    adj_ = adj + np.eye(adj.shape[0])
    
    # Calculate degree matrix
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.power(rowsum, -0.5).flatten()
    degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.
    degree_mat_inv_sqrt = np.diag(degree_mat_inv_sqrt)
    
    # Normalize adjacency matrix
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    
    # Convert to sparse tensor
    sparse_adj = coo_matrix(adj_normalized)
    indices = torch.from_numpy(np.vstack((sparse_adj.row, sparse_adj.col))).long()
    values = torch.from_numpy(sparse_adj.data).float()
    shape = torch.Size(sparse_adj.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

def save_mapping(mapping, output_file):
    """
    Save the computed node mapping to a file.
    """
    with open(output_file, "w") as f:
        for i, j in tqdm(enumerate(mapping), desc="Saving node mapping", total=len(mapping)):
            f.write(f"{i} -> {j.item()}\n")
    print(f"Node mapping saved to {output_file}")
