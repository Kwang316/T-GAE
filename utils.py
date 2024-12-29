import torch
import numpy as np
from scipy.sparse import coo_matrix
from netrd.distance import netsimile
import networkx as nx


def load_adj(dataset_path):
    return torch.load(dataset_path)


def test_matching(TGAE, S_hat_samples, p_samples, S_hat_features, S_emb, device, algorithm, metric):
    results = []
    for i in range(len(S_hat_samples)):
        S_hat_cur = S_hat_samples[i]
        adj_norm = preprocess_graph(coo_matrix(S_hat_cur.numpy()))
        adj_norm = torch.sparse.FloatTensor(
            torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])
        ).to(device)

        initial_feature = S_hat_features[i].to(device)
        z = TGAE(initial_feature, adj_norm).detach()
        D = torch.cdist(S_emb, z, p=2)

        if algorithm == "greedy":
            P_HG = greedy_hungarian(D, device)
        else:
            raise NotImplementedError("Only greedy matching is currently supported.")

        c = sum(P_HG[j].cpu().equal(p_samples[i][j].cpu()) for j in range(P_HG.size(0)))
        results.append(c / S_emb.shape[0])

    avg = np.average(results)
    std = np.std(results)
    return avg, std


def preprocess_graph(adj):
    adj = coo_matrix(adj)
    adj_ = adj + coo_matrix(np.eye(adj.shape[0]))
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = coo_matrix(np.diag(np.power(rowsum, -0.5).flatten()))
    return sparse_to_tuple(adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo())


def sparse_to_tuple(sparse_mx):
    if not isinstance(sparse_mx, coo_matrix):
        sparse_mx = sparse_mx.tocoo()
    return np.vstack((sparse_mx.row, sparse_mx.col)).transpose(), sparse_mx.data, sparse_mx.shape
