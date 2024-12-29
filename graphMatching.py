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




def fit_TGAE(model, train_loader, device, lr, epochs, level, eval_interval):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_avg = 0
    best_std = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for dataset, adj_list in train_loader.items():
            for adj in adj_list:
                adj = adj.to(device)
                adj_norm = preprocess_graph(adj)
                adj_norm = torch.sparse.FloatTensor(
                    torch.LongTensor(adj_norm[0].T),
                    torch.FloatTensor(adj_norm[1]),
                    torch.Size(adj_norm[2])
                ).to(device)

                optimizer.zero_grad()

                # Forward pass
                z = model(adj_norm)
                reconstructed = torch.sigmoid(torch.matmul(z, z.T))
                loss = F.binary_cross_entropy(reconstructed, adj_norm.to_dense())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if epoch % eval_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
            avg, std = test_matching(model, train_loader, level, device, "greedy", "accuracy")
            if avg > best_avg:
                best_avg = avg
                best_std = std
            print(f"Accuracy: {avg:.4f} ± {std:.4f}")

    print(f"Best Accuracy: {best_avg:.4f} ± {best_std:.4f}")
    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        print("Training mode activated...")
        adj = load_adj(args.dataset1)
        print("Adjacency matrix loaded.")

        train_loader = {"dataset1": generate_purturbations(device, adj, args.level, 10, args.model)}
        print("Perturbations generated.")

        model = TGAE(args.hidden_layers, args.input_dim, args.hidden_dim, args.output_dim).to(device)
        print("Model initialized.")

        fit_TGAE(model, train_loader, device, args.lr, args.epochs, args.level, args.eval_interval)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    elif args.mode == "map":
        adj1 = load_adj(args.dataset1)
        adj2 = load_adj(args.dataset2)
        print("Loaded adjacency matrices.")

        model = TGAE(args.hidden_layers, args.input_dim, args.hidden_dim, args.output_dim).to(device)
        model.load_state_dict(torch.load(args.load_model))
        print("Model loaded.")

        # Perform mapping (using test_matching for accuracy)
        _, _ = test_matching(model, {"dataset1": [adj1], "dataset2": [adj2]}, args.level, device, args.algorithm, "accuracy")
        print("Mapping completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "map"], help="Mode: train or map")
    parser.add_argument("--dataset1", type=str, required=True, help="Path to the first dataset")
    parser.add_argument("--dataset2", type=str, help="Path to the second dataset (for mapping)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--save_model", type=str, help="Path to save the trained model")
    parser.add_argument("--load_model", type=str, help="Path to load a pre-trained model")
    parser.add_argument("--level", type=float, default=0.0, help="Perturbation level")
    parser.add_argument("--model", type=str, default="uniform", choices=["uniform", "degree"], help="Perturbation model")
    parser.add_argument("--algorithm", type=str, default="greedy", choices=["greedy", "exact"], help="Matching algorithm")
    parser.add_argument("--hidden_layers", type=int, default=8, help="Number of hidden layers")
    parser.add_argument("--input_dim", type=int, required=True, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, nargs="+", required=True, help="Hidden layer dimensions")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension")
    args = parser.parse_args()
    main(args)
