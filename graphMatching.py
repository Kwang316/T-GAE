import torch
from utils import load_adj, preprocess_graph, save_mapping
from torch_geometric.nn import GINConv
import torch.nn as nn
import argparse
from scipy.sparse import coo_matrix

def adj_to_edge_index(adj):
    """
    Converts a sparse adjacency matrix (COO format) to edge_index format.
    """
    if not isinstance(adj, coo_matrix):
        adj = coo_matrix(adj.cpu().numpy())
    edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
    return edge_index

class TGAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(TGAE_Encoder, self).__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim[0])
        self.convs = nn.ModuleList()

        if len(hidden_dim) != num_hidden_layers + 1:
            raise ValueError(f"hidden_dim list length ({len(hidden_dim)}) must be num_hidden_layers + 1 ({num_hidden_layers + 1})")

        for i in range(num_hidden_layers):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                        nn.ReLU(),
                        nn.Linear(hidden_dim[i + 1], hidden_dim[i + 1])
                    )
                )
            )
        total_hidden_dim = sum(hidden_dim)
        self.out_proj = nn.Linear(total_hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.in_proj(x)
        hidden_states = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            hidden_states.append(x)

        x = torch.cat(hidden_states, dim=1)
        x = self.out_proj(x)
        return x

class TGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(TGAE, self).__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

def fit_TGAE(model, adj, features, device, lr, epochs):
    edge_index = adj_to_edge_index(preprocess_graph(adj)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(features, edge_index)
        reconstructed = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss = nn.BCELoss()(reconstructed, adj.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model

def compute_mapping(model, adj1, adj2, device):
    edge_index1 = adj_to_edge_index(preprocess_graph(adj1)).to(device)
    edge_index2 = adj_to_edge_index(preprocess_graph(adj2)).to(device)
    features1 = torch.ones((adj1.shape[0], 1), device=device)
    features2 = torch.ones((adj2.shape[0], 1), device=device)

    embeddings1 = model(features1, edge_index1)
    embeddings2 = model(features2, edge_index2)
    similarities = torch.matmul(embeddings1, embeddings2.T)
    return torch.argmax(similarities, dim=1)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mapping_only:
        print("Loading trained model...")
        hidden_dim = [16] * (args.num_hidden_layers + 1)
        model = TGAE(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=8,
            num_hidden_layers=args.num_hidden_layers
        ).to(device)
        model.load_state_dict(torch.load(args.load_model))
        model.eval()

        adj1, _ = load_adj(args.dataset1)
        adj2, _ = load_adj(args.dataset2)
        mapping = compute_mapping(model, adj1, adj2, device)
        save_mapping(mapping, "node_mapping.txt")
    else:
        print("Loading dataset...")
        adj, _ = load_adj(args.dataset1)
        features = torch.ones((adj.shape[0], 1))

        print("Training model...")
        hidden_dim = [16] * (args.num_hidden_layers + 1)
        model = TGAE(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=8,
            num_hidden_layers=args.num_hidden_layers
        ).to(device)

        model = fit_TGAE(model, adj, features, device, args.lr, args.epochs)
        torch.save(model.state_dict(), "tgae_model.pt")
        torch.save(model.state_dict(), "/content/drive/My Drive/Neuro/TGAE/tgae_model.pt")
        print("Model saved as tgae_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TGAE for graph matching task.")
    parser.add_argument('--dataset1', type=str, required=True, help='Path to the first dataset')
    parser.add_argument('--dataset2', type=str, default=None, help='Path to the second dataset (for mapping only)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--mapping_only', action='store_true', help='Perform mapping only')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a trained model checkpoint')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='Number of hidden layers')
    args = parser.parse_args()
    main(args)
