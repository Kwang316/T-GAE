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
