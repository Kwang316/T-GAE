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

def test_matching(TGAE, S_hat_samples, p_samples, S_hat_features, S_emb, device, algorithm, metric):
    if (metric == "accuracy"):
        results = []
    else:
        results = {}
        results["hit@1"] = []
        results["hit@5"] = []
        results["hit@10"] = []
        results["hit@50"] = []
    for i in range(len(S_hat_samples)):
        S_hat_cur = S_hat_samples[i]
        adj = coo_matrix(S_hat_cur.numpy())
        adj_norm = preprocess_graph(adj)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2])).to(device)
        initial_feature = S_hat_features[i].to(device)
        z = TGAE(initial_feature, adj_norm).detach()
        D = torch.cdist(S_emb, z, p=2)
        if (metric == "accuracy"):
            if(algorithm == "greedy"):
                P_HG = greedy_hungarian(D, device)
            elif(algorithm == "exact"):
                P_HG = hungarian(D)
            elif(algorithm == "approxNN"):
                P_HG = approximate_NN(S_emb,z)
            else:
                print("Matching algorithm undefined")
                exit()
            c = 0
            P = p_samples[i]
            for j in range(P_HG.size(0)):
                r1 = P_HG[j].cpu()
                r2 = P[j].cpu()
                if (r1.equal(r2)): c += 1
            results.append(c / S_emb.shape[0])
        else:
            P = p_samples[i].T
            hitAtOne = 0
            hitAtFive = 0
            hitAtTen = 0
            hitAtFifty = 0
            for j in range(P.size(0)):
                label = torch.nonzero(P)[j][1]
                dist_list = D[j]
                sorted_neighbors = torch.argsort(dist_list).cpu()
                for hit in range(50):
                    if (sorted_neighbors[hit].item() == label):
                        if (hit == 0):
                            hitAtOne += 1
                            hitAtFive += 1
                            hitAtTen += 1
                            hitAtFifty += 1
                            break
                        elif (hit <= 4):
                            hitAtFive += 1
                            hitAtTen += 1
                            hitAtFifty += 1
                            break
                        elif (hit <= 9):
                            hitAtTen += 1
                            hitAtFifty += 1
                            break
                        elif (hit <= 49):
                            hitAtFifty += 1
                            break
            results["hit@1"].append(hitAtOne)
            results["hit@5"].append(hitAtFive)
            results["hit@10"].append(hitAtTen)
            results["hit@50"].append(hitAtFifty)

    if (metric == "accuracy"):
        results = np.array(results)
        avg = np.average(results)
        std = np.std(results)
        return avg, std
    else:
        hitAtOne = np.average(np.array(results["hit@1"]))
        stdAtOne = np.std(np.array(results["hit@1"]))
        hitAtFive = np.average(np.array(results["hit@5"]))
        stdAtFive = np.std(np.array(results["hit@5"]))
        hitAtTen = np.average(np.array(results["hit@10"]))
        stdAtTen = np.std(np.array(results["hit@10"]))
        hitAtFifty = np.average(np.array(results["hit@50"]))
        stdAtFifty = np.std(np.array(results["hit@50"]))
        num_nodes = S_emb.shape[0]
        print("Hit@1: ", end="")
        print(str(hitAtOne / num_nodes)[:6] + "+-" + str(stdAtOne / num_nodes)[:6])
        print("Hit@5: ", end="")
        print(str(hitAtFive / num_nodes)[:6] + "+-" + str(stdAtFive / num_nodes)[:6])
        print("Hit@10: ", end="")
        print(str(hitAtTen / num_nodes)[:6] + "+-" + str(stdAtTen / num_nodes)[:6])
        print("Hit@50: ", end="")
        print(str(hitAtFifty / num_nodes)[:6] + "+-" + str(stdAtFifty / num_nodes)[:6])
        print()

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
