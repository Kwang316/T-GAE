from model import TGAE  # Import your TGAE model
from utils import load_adj, load_features
import torch
import torch.nn as nn
import argparse

def fit_TGAE(model, adj, features, device, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # Normalize adjacency matrix
    adj = adj.float().to(device)
    adj = adj / adj.max()  # Ensure values are in [0, 1]
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(features, adj)
        loss = nn.BCEWithLogitsLoss()(reconstructed, adj)

        # Debugging outputs
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"Adjacency min: {adj.min().item()}, max: {adj.max().item()}")
        print(f"Reconstructed min: {reconstructed.min().item()}, max: {reconstructed.max().item()}")

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return model


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    adj = load_adj(args.dataset1)  # Load adjacency matrix
    features = load_features(args.dataset1)  # Load features

    # Update dimensions based on dataset
    input_dim = features.shape[1]
    hidden_dim = 16
    output_dim = adj.shape[0]
    num_hidden_layers = 3  # Set the number of hidden layers (adjust as needed)

    # Initialize model
    model = TGAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers
    )

    print("Training model...")
    model = fit_TGAE(model, adj, features, device, args.lr, args.epochs)

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TGAE for graph matching")
    parser.add_argument('--dataset1', type=str, required=True, help="Path to the first dataset (adjacency matrix and features)")
    parser.add_argument('--dataset2', type=str, required=False, help="Path to the second dataset for mapping (optional)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--mapping_only', action='store_true', help="Run in mapping-only mode")
    parser.add_argument('--load_model', type=str, help="Path to the pre-trained model to load")

    args = parser.parse_args()
    main(args)
