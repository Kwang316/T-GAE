from algorithm import *
from model import *
import torch
import argparse
from utils import load_adj, generate_purturbations, test_matching
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")


def fit_TGAE(model, train_loader, features, device, lr, epochs, level, eval_interval):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_avg, best_std = 0, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for adj, feature in zip(train_loader, features):
            adj = adj.to(device)
            feature = feature.to(device)
            optimizer.zero_grad()

            # Forward pass
            z = model(feature, adj)
            A_pred = torch.sigmoid(torch.matmul(z, z.t()))
            loss = torch.nn.BCELoss()(A_pred, adj.to_dense())

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        if epoch % eval_interval == 0:
            print(f"Evaluating at level {level}")
            avg, std = test_matching(model, train_loader, level, device, "greedy", "accuracy")
            if avg > best_avg:
                best_avg, best_std = avg, std
                print(f"New best result: {best_avg:.4f} ± {best_std:.4f}")


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


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.mode == "train":
        adj = load_adj(args.dataset1)
        features = generate_purturbations(device, adj, args.level, 10, args.model)
        model = TGAE(args.hidden_layers, args.input_dim, args.hidden_dim, args.output_dim).to(device)
        fit_TGAE(model, [adj], [features], device, args.lr, args.epochs, args.level, args.eval_interval)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")
    elif args.mode == "map":
        adj1, adj2 = load_adj(args.dataset1), load_adj(args.dataset2)
        features1, features2 = generate_features(adj1), generate_features(adj2)
        model = TGAE(args.hidden_layers, args.input_dim, args.hidden_dim, args.output_dim).to(device)
        model.load_state_dict(torch.load(args.load_model))
        mapping = map_datasets(model, adj1, adj2, features1, features2, device, args.algorithm)
        with open(args.save_mapping, 'w') as f:
            f.write('\n'.join([f"{src}->{tgt}" for src, tgt in mapping]))
        print(f"Mapping saved to {args.save_mapping}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'map'], help="Mode: 'train' to train the model, 'map' to map nodes")
    parser.add_argument('--dataset1', type=str, required=True, help="Path to the first dataset (adjacency matrix and features)")
    parser.add_argument('--dataset2', type=str, default=None, help="Path to the second dataset for mapping (required for 'map' mode)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--eval_interval', type=int, default=1, help="Interval for evaluation during training")
    parser.add_argument('--save_model', type=str, help="Path to save the trained model")
    parser.add_argument('--load_model', type=str, help="Path to load the trained model for mapping")
    parser.add_argument('--save_mapping', type=str, help="Path to save the node mapping")
    parser.add_argument('--hidden_layers', type=int, default=8, help="Number of hidden layers in the TGAE model")
    parser.add_argument('--input_dim', type=int, required=True, help="Input feature dimension")
    parser.add_argument('--hidden_dim', type=int, nargs='+', required=True, help="Dimensions of hidden layers")
    parser.add_argument('--output_dim', type=int, required=True, help="Output feature dimension")
    parser.add_argument('--level', type=float, default=0.0, help="Perturbation level for generating data")
    parser.add_argument('--model', type=str, default="uniform", choices=["uniform", "degree"], help="Perturbation model")
    parser.add_argument('--algorithm', type=str, default="greedy", choices=["greedy", "exact"], help="Matching algorithm")
    return parser.parse_args()
