from tqdm import tqdm
from model import TGAE
from utils import *
import torch
from torch.optim import Adam
import argparse

def fit_TGAE(TGAE, adj, features, device, lr, epochs):
    adj_norm = preprocess_graph(adj.numpy())
    adj_norm_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2])
    ).to(device)
    optimizer = Adam(TGAE.parameters(), lr=lr, weight_decay=5e-4)
    features = features.to(device)

    for epoch in tqdm(range(epochs), desc="Training TGAE"):
        TGAE.train()
        optimizer.zero_grad()
        embeddings = TGAE(features, adj_norm_tensor)
        reconstructed = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss = torch.nn.BCELoss()(reconstructed, adj.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return TGAE

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading datasets...")
    adj, node_mapping = load_graph_from_csv(args.dataset1)
    features = generate_features(adj)

    model = TGAE(input_dim=1, hidden_dims=[16] * 8, output_dim=8).to(device)
    print("Starting training...")
    model = fit_TGAE(model, adj, features, device, args.lr, args.epochs)

    torch.save(model.state_dict(), "tgae_model.pt")
    print("Model training complete and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGAE Training")
    parser.add_argument('--dataset1', type=str, required=True, help='Path to the first dataset CSV')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    main(args)
