import pandas as pd
import os
import torch
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from netrd.distance import netsimile

def load_graph_from_csv(csv_path, node_mapping=None):
    data = pd.read_csv(csv_path)
    if node_mapping is None:
        all_nodes = pd.concat([data['From Node ID'], data['To Node Id']]).unique()
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}

    data['From Node ID'] = data['From Node ID'].map(node_mapping)
    data['To Node Id'] = data['To Node Id'].map(node_mapping)

    num_nodes = len(node_mapping)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for _, row in data.iterrows():
        adj_matrix[row['From Node ID'], row['To Node Id']] = row['Weight']

    return adj_matrix, node_mapping

def preprocess_graph(adj):
    adj = coo_matrix(adj)
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.tensor(adj_normalized, dtype=torch.float)

def generate_features(adj_matrix):
    degree = torch.sum(adj_matrix, dim=1)
    return degree.unsqueeze(1)

def save_tensors(adj_matrix, features, output_path):
    torch.save({'adj_matrix': adj_matrix, 'features': features}, output_path)

def main():
    male_csv_path = os.path.join(data_path, "male_connectome_graph.csv")
    female_csv_path = os.path.join(data_path, "female_connectome_graph.csv")
    male_csv = pd.read_csv(male_csv_path)
    female_csv = pd.read_csv(female_csv_path)

    male_adj, node_mapping = load_graph_from_csv(male_csv)
    male_features = generate_features(male_adj)
    male_adj_norm = preprocess_graph(male_adj)
    save_tensors(male_adj_norm, male_features, "male_embeddings.pt")

    female_adj, _ = load_graph_from_csv(female_csv, node_mapping=node_mapping)
    female_features = generate_features(female_adj)
    female_adj_norm = preprocess_graph(female_adj)
    save_tensors(female_adj_norm, female_features, "female_embeddings.pt")

    print("Embeddings generated and saved successfully.")

if __name__ == "__main__":
    main()
