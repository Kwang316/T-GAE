def map_graphs(model, dataset1, dataset2, device, save_mapping_path=None):
    """
    Map two graphs using the trained model.

    Args:
        model (TGAE): The trained TGAE model.
        dataset1 (str): Path to the first dataset.
        dataset2 (str): Path to the second dataset.
        device (torch.device): Device for computation.
        save_mapping_path (str): Path to save the mapping results (optional).

    Returns:
        None
    """
    print("Loading datasets for mapping...")
    adj1 = load_adj(dataset1).to(device)
    adj2 = load_adj(dataset2).to(device)

    features1 = load_features(dataset1).to(device)
    features2 = load_features(dataset2).to(device)

    model.eval()
    with torch.no_grad():
        # Generate embeddings
        embedding1 = model.encoder(features1, adj1)
        embedding2 = model.encoder(features2, adj2)

        # Compute pairwise similarity (e.g., cosine similarity)
        similarity_matrix = torch.mm(embedding1, embedding2.T)
        mapping = torch.argmax(similarity_matrix, dim=1)

        print("Mapping completed.")
        print(f"Mapping result: {mapping.cpu().numpy()}")

        if save_mapping_path:
            torch.save(mapping.cpu(), save_mapping_path)
            print(f"Mapping saved to {save_mapping_path}")


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model = TGAE(input_dim=0, hidden_dim=16, output_dim=0, num_hidden_layers=3)  # Adjust as necessary
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)
    print("Model loaded successfully.")

    if args.mapping_only:
        print("Running mapping only...")
        map_graphs(
            model=model,
            dataset1=args.dataset1,
            dataset2=args.dataset2,
            device=device,
            save_mapping_path=args.save_mapping
        )
        return

    print("Training is disabled for --mapping_only mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TGAE for graph matching")
    parser.add_argument('--dataset1', type=str, required=True, help="Path to the first dataset (adjacency matrix and features)")
    parser.add_argument('--dataset2', type=str, required=False, help="Path to the second dataset for mapping (optional)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--mapping_only', action='store_true', help="Run in mapping-only mode")
    parser.add_argument('--load_model', type=str, required=True, help="Path to the pre-trained model to load")
    parser.add_argument('--save_mapping', type=str, help="Path to save the mapping result")

    args = parser.parse_args()
    main(args)
