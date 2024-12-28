import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
import numpy as np
from scipy.sparse import coo_matrix

def preprocess_graph(adj):
    """
    Normalize the adjacency matrix.
    """
    # Convert PyTorch tensor to numpy if needed
    if torch.is_tensor(adj):
        adj = adj.cpu().numpy()
    elif isinstance(adj, (coo_matrix, csr_matrix)):
        adj = adj.toarray()
    elif not isinstance(adj, np.ndarray):
        adj = np.array(adj)
        
    # Add self-loops
    adj_ = adj + np.eye(adj.shape[0])
    
    # Calculate degree matrix
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.power(rowsum, -0.5).flatten()
    degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.
    degree_mat_inv_sqrt = np.diag(degree_mat_inv_sqrt)
    
    # Normalize adjacency matrix
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    
    # Convert to sparse tensor
    sparse_adj = coo_matrix(adj_normalized)
    indices = torch.from_numpy(np.vstack((sparse_adj.row, sparse_adj.col))).long()
    values = torch.from_numpy(sparse_adj.data).float()
    shape = torch.Size(sparse_adj.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

# Rest of the code remains the same...
class TGAE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(TGAE_Encoder, self).__init__()
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim[0])
        
        # Create ModuleList for GIN layers
        self.convs = torch.nn.ModuleList()
        
        # Ensure hidden_dim list has correct length
        if len(hidden_dim) != num_hidden_layers + 1:
            raise ValueError(f"hidden_dim list length ({len(hidden_dim)}) must be num_hidden_layers + 1 ({num_hidden_layers + 1})")
            
        # Create GIN layers
        for i in range(num_hidden_layers):
            self.convs.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim[i+1], hidden_dim[i+1])
                    )
                )
            )
            
        # Output projection
        total_hidden_dim = sum(hidden_dim)  # Sum of all hidden dimensions for concatenation
        self.out_proj = torch.nn.Linear(total_hidden_dim, output_dim)

    def forward(self, x, adj):
        # Initial projection
        x = self.in_proj(x)
        hidden_states = [x]
        
        # Apply GIN layers and collect hidden states
        for conv in self.convs:
            x = conv(x, adj)
            hidden_states.append(x)
            
        # Concatenate all hidden states
        x = torch.cat(hidden_states, dim=1)
        
        # Final projection
        x = self.out_proj(x)
        return x

class TGAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(TGAE, self).__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers)

    def forward(self, x, adj):
        z = self.encoder(adj, x)
        return z

def fit_TGAE(model, adj, features, device, lr, epochs):
    """
    Train the TGAE model.
    """
    adj_norm = preprocess_graph(adj)
    adj_norm = adj_norm.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    features = features.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(features, adj_norm)
        reconstructed = torch.sigmoid(torch.matmul(embeddings, embeddings.T))
        loss = torch.nn.BCELoss()(reconstructed, adj.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.mapping_only:
        print("Loading trained model...")
        # Define model with correct dimensions
        hidden_dim = [16] * (args.num_hidden_layers + 1)  # Add one extra element for output
        model = TGAE(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=8,
            num_hidden_layers=args.num_hidden_layers
        ).to(device)
        
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
        features = torch.ones((adj.shape[0], 1))  # Generate dummy features

        print("Training model...")
        hidden_dim = [16] * (args.num_hidden_layers + 1)  # Add one extra element for output
        model = TGAE(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=8,
            num_hidden_layers=args.num_hidden_layers
        ).to(device)
        a
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
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='Number of hidden layers')
    args = parser.parse_args()
    main(args)
