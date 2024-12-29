from model import TGAE
from utils import load_adj, load_features
import torch
import argparse
from torch.optim import Adam
import warnings

warnings.filterwarnings("ignore")


def fit_TGAE(model, adj, features, device, lr, epochs, save_model_path):
    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)
    adj = adj.to(device)
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(features, adj)
        reconstruction = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss = torch.nn.BCELoss()(reconstruction, adj)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")


def map_nodes(model, dataset1_adj, dataset1_features, dataset2_adj, dataset2_features, device, save_mapping_path):
    model.to(device)
    model.eval()

    dataset1_adj = dataset1_adj.to(device)
    dataset1_features = dataset1_features.to(device)
    dataset2_adj = dataset2_adj.to(device)
    dataset2_features = dataset2_features.to(device)

    # Generate embeddings for both datasets
    emb1 = model(dataset1_features, dataset1_adj)
    emb2 = model(dataset2_features, dataset2_adj)

    # Compute pairwise distances
    distances = torch.cdist(emb1, emb2, p=2)

    # Compute node mapping using Hungarian algorithm
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(distances.cpu().detach().numpy())
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}

    if save_mapping_path:
        torch.save(mapping, save_mapping_path)
        print(f"Node mapping saved to {save_mapping_path}")

    return mapping


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        print("Loading dataset...")
        adj = load_adj(args.dataset1)
        features = load_features(args.dataset1)

        input_dim = features.shape[1]
        hidden_dim = [16] * 8  # 8 hidden layers
        output_dim = 16  # Output feature size

        model = TGAE(len(hidden_dim), input_dim, hidden_dim, output_dim)

        print("Training model...")
        fit_TGAE(
            model,
            adj,
            features,
            device,
            args.lr,
            args.epochs,
            args.save_model,
        )

    elif args.mode == "map":
        if not args.dataset2 or not args.load_model:
            raise ValueError("Mapping mode requires --dataset2 and --load_model")

        print("Loading datasets...")
        dataset1_adj = load_adj(args.dataset1)
        dataset1_features = load_features(args.dataset1)
        dataset2_adj = load_adj(args.dataset2)
        dataset2_features = load_features(args.dataset2)

        print("Loading model...")
        input_dim = dataset1_features.shape[1]
        hidden_dim = [16] * 8
        output_dim = 16
        model = TGAE(len(hidden_dim), input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(args.load_model))

        print("Mapping nodes...")
        map_nodes(
            model,
            dataset1_adj,
            dataset1_features,
            dataset2_adj,
            dataset2_features,
            device,
            args.save_mapping,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument("--mode", type=str, choices=["train", "map"], required=True, help="Mode: train or map")
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first dataset")
    parser.add_argument("--dataset2", type=str, help="Path to the second dataset for mapping")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--save_model", type=str, help="Path to save the trained model")
    parser.add_argument("--load_model", type=str, help="Path to load the trained model")
    parser.add_argument("--save_mapping", type=str, help="Path to save the node mapping")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
