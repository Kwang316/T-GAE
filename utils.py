import torch


def load_adj(filepath):
    return torch.load(filepath)


def generate_purturbations(device, adj, level, samples, method="uniform"):
    perturbed = []
    for _ in range(samples):
        if method == "uniform":
            # Example perturbation logic
            perturbed.append(adj)
    return perturbed


def test_matching(z1, z2, algorithm, metric="accuracy"):
    if algorithm == "greedy":
        return [(i, j) for i, j in zip(range(z1.size(0)), range(z2.size(0)))]
    else:
        raise NotImplementedError("Algorithm not supported.")
