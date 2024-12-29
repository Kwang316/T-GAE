def load_adj(filepath):
    """
    Load an adjacency matrix from a .pt file.

    Args:
        filepath (str): Path to the adjacency matrix file.

    Returns:
        torch.Tensor: Dense adjacency matrix as a PyTorch tensor.
    """
    data = torch.load(filepath)
    
    if isinstance(data, dict):
        if "adj_matrix" in data:
            return data["adj_matrix"]  # Extract the adjacency matrix
        else:
            raise ValueError(f"File {filepath} does not contain 'adj_matrix' key.")
    
    elif isinstance(data, torch.Tensor):
        return data  # Return directly if it's a tensor

    else:
        raise ValueError(f"File {filepath} does not contain a valid adjacency matrix.")
