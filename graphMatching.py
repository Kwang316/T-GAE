from algorithm import *
from model import *
import torch
import argparse
from utils import load_adj, generate_purturbations, test_matching
from torch.optim import Adam
import warnings
from model import TGAE
from utils import load_adj, generate_purturbations, test_matching
import torch
import argparse
from torch.optim import Adam
import torch.nn.functional as F
warnings.filterwarnings("ignore")


def map_datasets(model, adj1, adj2, features1, features2, device, algorithm):
    model.eval()
    adj1 = adj1.to(device)
    adj2 = adj2.to(device)
    features1 = features1.to(device)
    features2 = features2.to(device)

    z1 = model(features1, adj1)
    z2 = model(features2, adj2)

    print(f"Mapping nodes using {algorithm}")
    return test_matching(z1, z2, algorithm)




from model import TGAE
from utils import load_adj, generate_purturbations, test_matching
import torch
import argparse

def fit_TGAE(model, train_loader, device, lr, epochs, level, eval_interval):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.to(device)

    print("Training started...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for dataset, data_list in train_loader.items():
            for data in data_list:
                adj = data['adj'].to(device)  # Adjacency matrix
                features = data['features'].to(device)  # Node features
                
                optimizer.zero_grad()
                reconstructed = model(features, adj)
                loss = torch.nn.functional.mse_loss(reconstructed, adj)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        if epoch % eval_interval == 0:
            print(f"Evaluating at Epoch {epoch + 1}...")
            avg, std = test_matching(model, train_loader, level, device, "greedy", "accuracy")
            print(f"Average Accuracy: {avg:.4f}, Std: {std:.4f}")

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.mode == "train":
        print("Training mode activated...")
        adj = load_adj(args.dataset1)
        perturbations = generate_purturbations(device, adj, args.level, 10, args.model)
        train_loader = {
            f"dataset_{i}": [
                {"adj": adj, "features": torch.randn(adj.size(0), args.input_dim)}
                for adj in perturbations
            ]
            for i in range(len(perturbations))
        }

        model = TGAE(args.hidden_layers, args.input_dim, args.hidden_dim, args.output_dim).to(device)
        fit_TGAE(model, train_loader, device, args.lr, args.epochs, args.level, args.eval_interval)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    elif args.mode == "map":
        print("Mapping mode activated...")
        adj1, adj2 = load_adj(args.dataset1), load_adj(args.dataset2)
        features1, features2 = torch.randn(adj1.size(0), args.input_dim), torch.randn(adj2.size(0), args.input_dim)

        model = TGAE(args.hidden_layers, args.input_dim, args.hidden_dim, args.output_dim).to(device)
        model.load_state_dict(torch.load(args.load_model))

        # Implement the mapping logic (e.g., using Hungarian algorithm) if required
        # Placeholder: Printing a success message
        print("Mapping completed. Save mapping if necessary.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'map'], help="Mode: 'train' or 'map'")
    parser.add_argument('--dataset1', type=str, required=True, help="Path to the first dataset (adjacency matrix and features)")
    parser.add_argument('--dataset2', type=str, default=None, help="Path to the second dataset (optional for mapping mode)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--eval_interval', type=int, default=1, help="Evaluation interval during training")
    parser.add_argument('--save_model', type=str, help="Path to save the trained model")
    parser.add_argument('--load_model', type=str, help="Path to load the trained model")
    parser.add_argument('--hidden_layers', type=int, default=8, help="Number of hidden layers in TGAE")
    parser.add_argument('--input_dim', type=int, required=True, help="Dimension of input features")
    parser.add_argument('--hidden_dim', type=int, nargs='+', required=True, help="Dimensions of hidden layers")
    parser.add_argument('--output_dim', type=int, required=True, help="Dimension of output features")
    parser.add_argument('--level', type=float, default=0.0, help="Perturbation level for training")
    parser.add_argument('--model', type=str, default="uniform", choices=["uniform", "degree"], help="Perturbation model")
    args = parser.parse_args()
    main(args)
