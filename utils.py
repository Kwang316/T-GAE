import torch
import numpy as np
from scipy.sparse import coo_matrix
import random

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

def gen_dataset(adj, num_to_add, num_to_delete):
    size = adj.shape[0]
    adj = adj.clone()
    edges = adj.nonzero(as_tuple=False).tolist()
    non_edges = [(i, j) for i in range(size) for j in range(size) if i != j and adj[i, j] == 0]

    # Remove edges
    for _ in range(min(num_to_delete, len(edges))):
        edge = random.choice(edges)
        adj[edge[0], edge[1]] = 0
        adj[edge[1], edge[0]] = 0
        edges.remove(edge)

    # Add edges
    for _ in range(min(num_to_add, len(non_edges))):
        non_edge = random.choice(non_edges)
        adj[non_edge[0], non_edge[1]] = 1
        adj[non_edge[1], non_edge[0]] = 1
        non_edges.remove(non_edge)

    return adj

def sparse_to_tuple(sparse_mx):
    if not isinstance(sparse_mx, coo_matrix):
        sparse_mx = sparse_mx.tocoo()
    return np.vstack((sparse_mx.row, sparse_mx.col)).transpose(), sparse_mx.data, sparse_mx.shape
