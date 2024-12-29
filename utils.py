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


def generate_purturbations(device, S, perturbation_level, no_samples, method):
    purturbated_samples = []
    if(method == "uniform"):
        for i in range(no_samples):
            num_edges = int(torch.count_nonzero(S).item()/2)
            total_purturbations = int(perturbation_level * num_edges)
            add_edge = random.randint(0,total_purturbations)
            delete_edge = total_purturbations - add_edge
            S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
            purturbated_samples.append(S_prime)
    elif(method == "degree"):
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(perturbation_level * num_edges)
        S = torch.triu(S, diagonal=0)
        ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
        ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
        ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
        try:
            D = S @ ones_long
        except:
            try:
                D = S @ ones_int
            except:
                D = S @ ones_float

        sum = torch.sum(torch.mul(D@D.T,S))
        edge_index = S.nonzero().t().contiguous()
        edge_index = np.array(edge_index)
        prob = []
        for i in range(edge_index.shape[1]):
            d1 = edge_index[0,i]
            d2 = edge_index[1,i]
            prob.append(D[d1]*D[d2]/sum)
        prob = np.array(prob,dtype='float64')
        prob = np.squeeze(prob)
        for i in range(no_samples):
            edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations,False,p=prob)
            edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
            edges_index = edge_index[:,edges_remain]
            S_prime = torch.zeros_like(S)
            for j in range(edges_index.shape[1]):
                n1 = edges_index[:,j][0]
                n2 = edges_index[:,j][1]
                S_prime[n1][n2] = 1
                S_prime[n2][n1] = 1
            purturbated_samples.append(S_prime)
    else:
        print("Probability model not defined.")
        exit()
    return purturbated_samples

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
