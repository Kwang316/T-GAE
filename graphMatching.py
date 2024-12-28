from utils import load_adj, preprocess_graph
import torch
from torch.optim import Adam
from model import TGAE
import argparse

def fit_TGAE(model, adj, features, device, lr, epochs):
    """
    Train the TGAE model.
    """
    adj_norm = preprocess_graph(adj.numpy())
    adj_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(features, adj_norm_tensor)
        reconstructed = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss = torch.nn.BCELoss()(reconstructed, adj.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model

def compute_mapping(model, adj1, adj2, device):
    """
    Compute node mappings between two graphs.
    """
    adj1 = preprocess_graph(adj1.numpy()).to(device)
    adj2 = preprocess_graph(adj2.numpy()).to(device)
    features1 = torch.ones((adj1.shape[0], 1), device=device)
    features2 = torch.ones((adj2.shape[0], 1), device=device)

    embeddings1 = model(features1, adj1)
    embeddings2 = model(features2, adj2)

    # Compute pairwise similarities
    similarities = torch.matmul(embeddings1, embeddings2.T)
    mapping = torch.argmax(similarities, dim=1)  # Find the best match for each node

    return mapping

def save_mapping(mapping, output_file):
    """
    Save the computed node mapping to a file.
    """
    with open(output_file, "w") as f:
        for i, j in enumerate(mapping):
            f.write(f"{i} -> {j.item()}\n")
    print(f"Node mapping saved to {output_file}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mapping_only:
        print("Loading trained model...")
        model = TGAE(input_dim=1, hidden_dim=16, output_dim=8).to(device)
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        print("Loaded trained model. Computing mapping...")

        adj1, _ = load_adj(args.dataset1)
        adj2, _ = load_adj(args.dataset2)

        mapping = compute_mapping(model, adj1, adj2, device)
        save_mapping(mapping, "node_mapping.txt")
    else:
        print("Loading dataset...")
        adj, _ = load_adj(args.dataset1)
        features = torch.ones((adj.shape[0], 1))  # Generate dummy features (e.g., ones)

        print("Training model...")
        model = TGAE(input_dim=1, hidden_dim=16, output_dim=8).to(device)
        model = fit_TGAE(model, adj, features, device, args.lr, args.epochs)

        print("Saving trained model...")
        torch.save(model.state_dict(), "tgae_model.pt")
        print("Model saved as tgae_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TGAE for graph matching task.")
    parser.add_argument('--dataset1', type=str, required=True, help='Path to the first dataset')
    parser.add_argument('--dataset2', type=str, default=None, help='Path to the second dataset (for mapping only)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--mapping_only', action='store_true', help='Perform mapping only')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a trained model checkpoint')
    args = parser.parse_args()
    main(args)
