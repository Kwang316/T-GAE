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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        # Load dataset
        print("Loading dataset...")
        adj = load_adj(args.dataset1).to(device)

        # Model parameters
        input_dim = adj.shape[1]
        hidden_dim = [16, 16, 16, 16]
        output_dim = 16

        model = TGAE(len(hidden_dim), input_dim, hidden_dim, output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            reconstructed = model(adj, adj)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(reconstructed, adj)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Evaluation
            if epoch % args.eval_interval == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save the model
        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")

    elif args.mode == 'map':
        # Load datasets and pre-trained model
        print("Loading datasets...")
        adj1 = load_adj(args.dataset1).to(device)
        adj2 = load_adj(args.dataset2).to(device)

        print("Loading model...")
        model = TGAE(len(args.hidden_dim), adj1.shape[1], args.hidden_dim, 16).to(device)
        model.load_state_dict(torch.load(args.load_model))
        model.eval()

        # Map nodes
        print("Mapping nodes...")
        embeddings1 = model(adj1, adj1).detach()
        embeddings2 = model(adj2, adj2).detach()

        # Perform node matching
        print("Calculating node matching...")
        if args.algorithm == 'greedy':
            mapping = greedy_hungarian(torch.cdist(embeddings1, embeddings2))
        elif args.algorithm == 'exact':
            mapping = hungarian(torch.cdist(embeddings1, embeddings2))
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")

        print("Node mapping completed!")
        if args.save_mapping:
            torch.save(mapping, args.save_mapping)
            print(f"Node mapping saved to {args.save_mapping}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TGAE for graph matching task")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'map'], help="Mode: train or map")
    parser.add_argument('--dataset1', type=str, required=True, help="Path to the first dataset (adjacency matrix)")
    parser.add_argument('--dataset2', type=str, help="Path to the second dataset (for mapping)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--eval_interval', type=int, default=5, help="Evaluation interval during training")
    parser.add_argument('--save_model', type=str, help="Path to save the trained model")
    parser.add_argument('--load_model', type=str, help="Path to the pre-trained model")
    parser.add_argument('--save_mapping', type=str, help="Path to save the node mapping")
    parser.add_argument('--model', type=str, default='uniform', choices=['uniform', 'degree'], help="Perturbation model")
    parser.add_argument('--algorithm', type=str, default='greedy', choices=['greedy', 'exact'], help="Matching algorithm")

    args = parser.parse_args()
    main(args)

