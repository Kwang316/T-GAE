from algorithm import *
from model import *
from utils import *  # Ensure all utility functions, including generate_purturbations, are imported
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
import argparse

warnings.filterwarnings("ignore")


def fit_TGAE(no_samples, TGAE, epoch, train_loader, train_features, device, lr, level_eval, dataset_eval, model_eval, algorithm, eval_interval):
    """Fit the TGAE model."""
    best_avg = 0
    best_std = 0
    S_hat_samples, S_prime_samples, p_samples = gen_test_set(device, load_adj(dataset_eval), 10, [level_eval], method=model_eval)
    S_eval = load_adj(dataset_eval)
    adj_S = coo_matrix(S_eval.numpy())
    adj_norm_S = preprocess_graph(adj_S)
    adj_norm_S = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm_S[0].T),
        torch.FloatTensor(adj_norm_S[1]),
        torch.Size(adj_norm_S[2])
    ).to(device)
    S_feat = generate_features([S_eval])[0]

    S_hat_features = generate_features(S_hat_samples[str(level_eval)])
    optimizer = torch.optim.Adam(TGAE.parameters(), lr=lr, weight_decay=5e-4)
    for step in range(epoch):
        loss = 0
        for dataset in train_loader.keys():
            S = train_loader[dataset][0]
            initial_features = train_features[dataset]
            for i in range(len(train_loader[dataset])):
                adj_tensor = train_loader[dataset][i]
                adj = coo_matrix(adj_tensor.numpy())
                adj_norm = preprocess_graph(adj)
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
                adj_label = coo_matrix(S.numpy())
                adj_label = sparse_to_tuple(adj_label)
                adj_norm = torch.sparse.FloatTensor(
                    torch.LongTensor(adj_norm[0].T),
                    torch.FloatTensor(adj_norm[1]),
                    torch.Size(adj_norm[2])
                ).to(device)
                adj_label = torch.sparse.FloatTensor(
                    torch.LongTensor(adj_label[0].T),
                    torch.FloatTensor(adj_label[1]),
                    torch.Size(adj_label[2])
                ).to(device)

                initial_feature = initial_features[i].to(device)
                weight_mask = adj_label.to_dense().view(-1) == 1
                weight_tensor = torch.ones(weight_mask.size(0))
                weight_tensor[weight_mask] = pos_weight
                weight_tensor = weight_tensor.to(device)
                z = TGAE(initial_feature, adj_norm)
                A_pred = torch.sigmoid(torch.matmul(z, z.t()))
                loss += norm * torch.nn.functional.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                                         weight=weight_tensor)
        optimizer.zero_grad()
        loss = loss / no_samples
        loss.backward()
        optimizer.step()

        S_emb = TGAE(S_feat.to(device), adj_norm_S).detach()
        if step % eval_interval == 0:
            print(f"Epoch: {step + 1}, train_loss= {loss.item():.5f}", end=" ")
            avg, std = test_matching(TGAE, S_hat_samples[str(level_eval)], p_samples[str(level_eval)],
                                     S_hat_features, S_emb, device, algorithm, metric="accuracy")
            if avg > best_avg:
                best_avg = avg
                best_std = std
            print(f"Current best result: {best_avg:.6f} ± {best_std:.5f}")
        else:
            print(f"Epoch: {step + 1}, train_loss= {loss.item():.5f}")

    return TGAE


def map_nodes(TGAE, dataset1, dataset2, device, algorithm, metric):
    """Map nodes between two datasets."""
    adj1 = load_adj(dataset1).to(device)
    adj2 = load_adj(dataset2).to(device)
    features1 = generate_features([adj1])[0].to(device)
    features2 = generate_features([adj2])[0].to(device)

    emb1 = TGAE(features1, adj1).detach()
    emb2 = TGAE(features2, adj2).detach()

    D = torch.cdist(emb1, emb2, p=2)
    if algorithm == "greedy":
        mapping = greedy_hungarian(D, device)
    elif algorithm == "exact":
        mapping = hungarian(D)
    else:
        raise ValueError("Unknown mapping algorithm.")
    
    # Save mapping results
    np.save("node_mapping.npy", mapping.cpu().numpy())
    print("Mapping saved as node_mapping.npy")

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    train_set = ["celegans", "arenas", "douban", "cora"]
    test_set = [args.dataset1]
    probability_model = args.model
    training_perturbation_level = args.level
    no_training_samples_per_graph = 10
    NUM_HIDDEN_LAYERS = 8
    HIDDEN_DIM = [16, 16, 16, 16, 16, 16, 16, 16, 16]
    output_feature_size = 1 if args.algorithm == "approxNN" else 8
    lr = args.lr
    epoch = args.epochs

    print("Loading training datasets")

    train_loader = {
        dataset: generate_purturbations(
            device, load_adj(dataset), training_perturbation_level, no_training_samples_per_graph, probability_model
        )
        for dataset in train_set
    }

    model = TGAE(NUM_HIDDEN_LAYERS, 7, HIDDEN_DIM, output_feature_size).to(device)

    print("Generating training features")
    train_features = {
        dataset: generate_features(train_loader[dataset]) for dataset in train_loader.keys()
    }

    print("Fitting T-GAE")
    fit_TGAE(
        len(train_set) * (no_training_samples_per_graph + 1),
        model,
        epoch,
        train_loader,
        train_features,
        device,
        lr,
        args.level,
        args.dataset1,
        args.model,
        args.algorithm,
        args.eval_interval,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument(
        "--mode", type=str, choices=["train", "map"], required=True, help="Choose between train or map mode"
    )
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first dataset")
    parser.add_argument("--dataset2", type=str, help="Path to the second dataset for mapping (required in map mode)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--level", type=float, default=0.0, help="Perturbation level for training datasets")
    parser.add_argument("--model", type=str, default="uniform", help="Probability model (uniform or degree)")
    parser.add_argument("--algorithm", type=str, default="greedy", help="Matching algorithm")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluation interval")
    parser.add_argument("--save_model", type=str, help="Path to save the trained model")
    parser.add_argument("--load_model", type=str, help="Path to load the trained model (for mapping)")
    parser.add_argument("--save_mapping", type=str, help="Path to save the node mapping (for mapping)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
