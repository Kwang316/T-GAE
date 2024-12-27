from algorithm import *
from model import *
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pickle
from netrd.distance import netsimile
import networkx as nx
import os.path as osp
from scipy.sparse import coo_matrix
from tqdm import tqdm
import random
import warnings
from torch.optim import Adam
from utils import *
import argparse
warnings.filterwarnings("ignore")
from algorithm import *
from model import *
import torch
import numpy as np
from utils import *
from torch.optim import Adam
import argparse


def fit_TGAE(TGAE, adj, features, device, lr, epochs):
    adj_norm = preprocess_graph(adj.numpy())
    adj_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
    ).to(device)
    optimizer = Adam(TGAE.parameters(), lr=lr, weight_decay=5e-4)
    features = features.to(device)

    for epoch in range(epochs):
        TGAE.train()
        optimizer.zero_grad()
        embeddings = TGAE(features, adj_norm_tensor)
        reconstructed = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss = torch.nn.BCELoss()(reconstructed, adj.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return TGAE


def compute_mapping(TGAE, adj1, features1, adj2, features2, device, node_mapping):
    adj1_norm = preprocess_graph(adj1.numpy())
    adj1_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj1_norm[0].T),
        torch.FloatTensor(adj1_norm[1]),
        torch.Size(adj1_norm[2])
    ).to(device)
    features1 = features1.to(device)
    adj2_norm = preprocess_graph(adj2.numpy())
    adj2_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj2_norm[0].T),
        torch.FloatTensor(adj2_norm[1]),
        torch.Size(adj2_norm[2])
    ).to(device)
    features2 = features2.to(device)

    TGAE.eval()
    embeddings1 = TGAE(features1, adj1_norm_tensor).detach()
    embeddings2 = TGAE(features2, adj2_norm_tensor).detach()

    similarities = torch.cdist(embeddings1, embeddings2, p=2)
    row_ind, col_ind = linear_sum_assignment(similarities.cpu().numpy())
    node_matching = [(node_mapping[row], node_mapping[col]) for row, col in zip(row_ind, col_ind)]

    with open("node_mapping.txt", "w") as f:
        for src, tgt in node_matching:
            f.write(f"{src} -> {tgt}\n")
    print("Node mapping saved to node_mapping.txt")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mapping_only:
        assert args.load_model, "Provide --load_model to compute mapping"
        model = TGAE(input_dim=1, hidden_dims=[16] * 8, output_dim=8).to(device)
        model.load_state_dict(torch.load(args.load_model))
        adj1, node_mapping = load_adj(args.dataset1)
        adj2, _ = load_adj(args.dataset2)
        features1 = generate_features([adj1])[0]
        features2 = generate_features([adj2])[0]
        compute_mapping(model, adj1, features1, adj2, features2, device, node_mapping)
    else:
        adj, node_mapping = load_adj(args.dataset1)
        features = generate_features([adj])[0]
        model = TGAE(input_dim=1, hidden_dims=[16] * 8, output_dim=8).to(device)
        model = fit_TGAE(model, adj, features, device, args.lr, args.epochs)
        torch.save(model.state_dict(), "tgae_model.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument('--dataset1', type=str, required=True, help='Path to first dataset CSV')
    parser.add_argument('--dataset2', type=str, help='Path to second dataset CSV for mapping')
    parser.add_argument('--mapping_only', action='store_true', help='Compute mapping only')
    parser.add_argument('--load_model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
