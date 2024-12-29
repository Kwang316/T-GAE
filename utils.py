import torch
import numpy as np
import scipy.sparse as sp
from netrd.distance import netsimile
import networkx as nx


def load_adj(dataset_path):
    """Load adjacency matrix."""
    adj = torch.load(dataset_path)
    return adj


def test_matching(TGAE, S_hat_samples, p_samples, S_hat_features, S_emb, device, algorithm, metric):
    """Evaluate node matching accuracy."""
    results = []
    for i in range(len(S_hat_samples)):
        S_hat_cur = S_hat_samples[i]
        adj = sp.coo_matrix(S_hat_cur.numpy())
        adj_norm = preprocess_graph(adj)
        adj_norm = torch.sparse.FloatTensor(
            torch.LongTensor(adj_norm[0].T),
            torch.FloatTensor(adj_norm[1]),
            torch.Size(adj_norm[2])
        ).to(device)
        initial_feature = S_hat_features[i].to(device)
        z = TGAE(initial_feature, adj_norm).detach()
        D = torch.cdist(S_emb, z, p=2)
        if algorithm == "greedy":
            P_HG = greedy_hungarian(D, device)
        elif algorithm == "exact":
            P_HG = hungarian(D)
        else:
            raise ValueError("Matching algorithm undefined.")
        c = sum([P_HG[j].cpu().equal(p_samples[i][j].cpu()) for j in range(P_HG.size(0))])
        results.append(c / S_emb.shape[0])
    avg = np.average(results)
    std = np.std(results)
    return avg, std


def preprocess_graph(adj):
    """Normalize graph adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
