import scipy
import torch
import numpy as np
import os.path as osp
from scipy.sparse import coo_matrix
import networkx as nx
from netrd.distance import netsimile
import random


def load_graph_from_csv(csv_path, node_mapping=None):
    import pandas as pd
    data = pd.read_csv(csv_path)

    # Create or reuse node mappings
    if node_mapping is None:
        all_nodes = pd.concat([data['From Node ID'], data['To Node Id']]).unique()
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}

    # Map node IDs to integer indices
    data['From Node ID'] = data['From Node ID'].map(node_mapping)
    data['To Node Id'] = data['To Node Id'].map(node_mapping)

    # Generate adjacency matrix
    num_nodes = len(node_mapping)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for _, row in data.iterrows():
        adj_matrix[row['From Node ID'], row['To Node Id']] = row['Weight']

    return adj_matrix, node_mapping


def load_adj(dataset):
    if dataset == "male_data":
        return load_graph_from_csv("male_data.csv")
    elif dataset == "female_data":
        return load_graph_from_csv("female_data.csv")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def preprocess_graph(adj):
    adj = coo_matrix(adj)
    adj_ = adj + scipy.sparse.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = scipy.sparse.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    if not scipy.sparse.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def generate_features(graphs):
    features = []
    for S in graphs:
        feature = gen_netsmile(S)
        features.append(feature)
    return features


def gen_netsmile(S):
    np_S = S.numpy()
    G = nx.from_numpy_array(np_S)
    feat = netsimile.feature_extraction(G)
    return torch.tensor(feat, dtype=torch.float)
