from model import TGAE  # Import your TGAE model
from utils import load_adj, load_features
import torch
import torch.nn as nn
import argparse


def fit_TGAE(model, adj, features, device, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    adj = (adj > 0).float().to(device)  # Ensure binary adjacency matrix
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(features, adj)

        # Ensure reconstructed values are in [0, 1]
        reconstructed = torch.sigmoid(reconstructed)

        # Compute loss
        loss = nn.BCELoss()(reconstructed, adj)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return model


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    adj = load_adj(args.dataset1)  # Load adjacency matrix
    features = load_features(args.features1)  # Load node features

    # Initialize model
    model = TGAE(encoder_hidden_dims=[32, 16], decoder_hidden_dims=[16, 32])

    print("Training model...")
    model = fit_TGAE(model, adj, features, device, args.lr, args.epochs)

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TGAE for graph matching")
    parser.add_argument('--dataset1', type=str, required=True, help="Path to the first dataset (adjacency matrix)")
    parser.add_argument('--features1', type=str, required=True, help="Path to the node features for the first dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    main(args)
