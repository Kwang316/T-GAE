from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from netrd.distance import netsimile

def preprocess_graph(adj):
    adj = coo_matrix(adj)
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.tensor(adj_normalized, dtype=torch.float)

def generate_features(adj_matrix):
    features = []
    for i, row in enumerate(tqdm(adj_matrix, desc="Generating features", total=adj_matrix.size(0))):
        degree = torch.sum(row)
        features.append(degree.item())
    return torch.tensor(features).unsqueeze(1)

def save_tensors(adj_matrix, features, output_path):
    torch.save({'adj_matrix': adj_matrix, 'features': features}, output_path)

def load_graph_from_csv(csv_path, node_mapping=None):
    data = pd.read_csv(csv_path)
    if node_mapping is None:
        all_nodes = pd.concat([data['From Node ID'], data['To Node Id']]).unique()
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}

    data['From Node ID'] = data['From Node ID'].map(node_mapping)
    data['To Node Id'] = data['To Node Id'].map(node_mapping)

    num_nodes = len(node_mapping)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for _, row in tqdm(data.iterrows(), desc="Building adjacency matrix", total=len(data)):
        adj_matrix[row['From Node ID'], row['To Node Id']] = row['Weight']

    return adj_matrix, node_mapping
