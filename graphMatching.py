from utils import load_adj, generate_purturbations, test_matching
from model import TGAE
import torch
import argparse
import os
from torch.optim import Adam
from tqdm import tqdm


def fit_TGAE(model, adj, features, device, lr, epochs, save_path=None):
    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)

    adj = adj.float().to(device)
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z = model(features, adj)
        A_pred = torch.sigmoid(torch.matmul(z, z.T))

        # Compute loss
        pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_tensor = torch.ones_like(adj)
        weight_tensor[adj == 1] = pos_weight
        loss = torch.nn.BCELoss(weight=weight_tensor)(A_pred.view(-1), adj.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model


def map_embeddings(model, dataset1, dataset2, device, algorithm="greedy"):
    adj1 = load_adj(dataset1).to(device)
    adj2 = load_adj(dataset2).to(device)
    features1 = torch.eye(adj1.shape[0]).to(device)
    features2 = torch.eye(adj2.shape[0]).to(device)

    model.eval()
    z1 = model(features1, adj1).detach()
    z2 = model(features2, adj2).detach()

    D = torch.cdist(z1, z2, p=2)
    mapping = test_matching(model, [adj1], [D], [features1], z2, device, algorithm, "accuracy")
    print(f"Mapping results: {mapping}")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        adj = load_adj(args.dataset1)
        features = torch.eye(adj.shape[0])  # Identity matrix as features

        input_dim = features.shape[1]
        hidden_dim = [16] * args.num_hidden_layers
        output_dim = 16

        model = TGAE(len(hidden_dim), input_dim, hidden_dim, output_dim)
        fit_TGAE(model, adj, features, device, args.lr, args.epochs, save_path=args.save_model)

    elif args.mode == "map":
        model = TGAE(args.num_hidden_layers, 16, [16] * args.num_hidden_layers, 16)
        model.load_state_dict(torch.load(args.load_model))
        model.to(device)

        map_embeddings(model, args.dataset1, args.dataset2, device, args.algorithm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument("--mode", choices=["train", "map"], required=True, help="Mode: train or map")
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first dataset")
    parser.add_argument("--dataset2", type=str, required=False, help="Path to the second dataset (required for mapping)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_model", type=str, help="Path to save the trained model")
    parser.add_argument("--load_model", type=str, help="Path to load the trained model")
    parser.add_argument("--num_hidden_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--algorithm", choices=["greedy", "exact", "approxNN"], default="greedy", help="Matching algorithm")

    args = parser.parse_args()
    main(args)
