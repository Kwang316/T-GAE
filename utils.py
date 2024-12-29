import torch

def load_adj(filepath):
    """
    Load an adjacency matrix from a .pt file.

    Args:
        filepath (str): Path to the adjacency matrix file.

    Returns:
        torch.Tensor: Dense adjacency matrix as a PyTorch tensor.
    """
    data = torch.load(filepath)
    
    if isinstance(data, dict) and "adj_matrix" in data:
        return data["adj_matrix"]
    else:
        raise ValueError(f"File {filepath} does not contain a valid adjacency matrix.")


def load_features(filepath):
    """
    Load node features from a .pt file.

    Args:
        filepath (str): Path to the file containing features.

    Returns:
        torch.Tensor: Node features as a PyTorch tensor.
    """
    data = torch.load(filepath)

    if isinstance(data, dict) and "features" in data:
        return data["features"]
    else:
        raise ValueError(f"File {filepath} does not contain 'features' key or valid node features.")
