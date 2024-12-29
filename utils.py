import torch
import random
import numpy as np

def load_adj(dataset_path):
    return torch.load(dataset_path)

def generate_purturbations(device, adj, perturbation_level, no_samples, method="uniform"):
    perturbed_samples = []
    num_edges = int(torch.count_nonzero(adj).item() / 2)
    total_perturbations = int(perturbation_level * num_edges)

    if method == "uniform":
        for _ in range(no_samples):
            add_edge = random.randint(0, total_perturbations)
            delete_edge = total_perturbations - add_edge
            perturbed_samples.append(gen_dataset(adj, add_edge, delete_edge))
    else:
        raise NotImplementedError("Only uniform perturbations are supported.")

    return perturbed_samples

def gen_dataset(adj, add_edge, delete_edge):
    # Placeholder for adjacency perturbation logic
    # Modify adjacency matrix here
    return adj



def test_matching(z1, z2, algorithm, metric="accuracy"):
    if algorithm == "greedy":
        return [(i, j) for i, j in zip(range(z1.size(0)), range(z2.size(0)))]
    else:
        raise NotImplementedError("Algorithm not supported.")
