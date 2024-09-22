import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GatedGCNConv
from sklearn.metrics import confusion_matrix, f1_score
import argparse
import sys
import numpy as np

# -----------------------------
# Model Definition
# -----------------------------

class EdgeClassifier(nn.Module):
    """
    Graph Neural Network model for edge classification using GatedGCNConv.

    Args:
        in_feats (int): Size of input node features.
        hidden_size (int): Size of hidden layers.
        num_classes (int): Number of classes for edge classification.
        edge_feat_size (int): Size of edge features.
        num_layers (int): Number of GatedGCNConv layers.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, in_feats, hidden_size, num_classes, edge_feat_size=1, num_layers=3, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        
        self.convs.append(
            GatedGCNConv(
                input_feats=in_feats,
                edge_feats=edge_feat_size,
                output_feats=hidden_size,
                activation=F.elu,
                dropout=dropout
            )
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GatedGCNConv(
                    input_feats=hidden_size,
                    edge_feats=hidden_size,
                    output_feats=hidden_size,
                    activation=F.leaky_relu,
                    dropout=dropout
                )
            )
        
        self.convs.append(
            GatedGCNConv(
                input_feats=hidden_size,
                edge_feats=hidden_size,
                output_feats=hidden_size,
                activation=F.leaky_relu,
                dropout=dropout
            )
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, g, node_feats, edge_feats):
        """
        Forward pass of the model.

        Args:
            g (dgl.DGLGraph): The input graph.
            node_feats (torch.Tensor): Node features of shape (N, in_feats).
            edge_feats (torch.Tensor): Edge features of shape (E, edge_feat_size).

        Returns:
            torch.Tensor: Logits for edge classification of shape (E, num_classes).
        """
        h = node_feats
        e = edge_feats

        for conv in self.convs:
            h, e = conv(g, h, e)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            e = F.leaky_relu(e)
            e = F.dropout(e, p=self.dropout, training=self.training)

        src_h = h[g.edges()[0]]
        dst_h = h[g.edges()[1]]
        edge_repr = torch.cat([src_h, dst_h, e], dim=1)  # [E, 3 * hidden_size]
        logits = self.edge_mlp(edge_repr)
        return logits

# -----------------------------
# Data Loading and Graph Construction
# -----------------------------

def load_csv_data(data_directory):
    """
    Loads node and edge data from CSV files specified in a metadata YAML file.

    Args:
        data_directory (str): Directory containing data and metadata files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing node and edge data.
    """
    meta_path = os.path.join(data_directory, 'meta.yaml')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)

    node_file = os.path.join(data_directory, meta['node_data'][0]['file_name'])
    edge_file = os.path.join(data_directory, meta['edge_data'][0]['file_name'])

    if not os.path.exists(node_file):
        raise FileNotFoundError(f"Node data file not found at {node_file}")
    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"Edge data file not found at {edge_file}")

    nodes_df = pd.read_csv(node_file)
    edges_df = pd.read_csv(edge_file)

    return nodes_df, edges_df


def construct_dgl_graphs(nodes_df, edges_df):
    """
    Constructs a list of DGL graphs from node and edge data.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node data.
        edges_df (pd.DataFrame): DataFrame containing edge data.

    Returns:
        List[dgl.DGLGraph]: List of DGL graphs.
    """
    graph_ids = nodes_df['graph_id'].unique()
    graphs = []

    for gid in graph_ids:
        nodes = nodes_df[nodes_df['graph_id'] == gid]
        edges = edges_df[edges_df['graph_id'] == gid]

        num_nodes = nodes['node_id'].nunique()
        node_ids = nodes['node_id'].tolist()
        node_id_map = {nid: idx for idx, nid in enumerate(node_ids)}

        src = edges['src_id'].map(node_id_map).tolist()
        dst = edges['dst_id'].map(node_id_map).tolist()

        # Assuming 'feat' is a single feature column; modify if multiple features
        node_feats = torch.tensor(nodes['feat'].astype(float).values).unsqueeze(1)
        edge_feats = torch.tensor(edges['feat'].astype(float).values).unsqueeze(1)
        labels = torch.tensor(edges['label'].astype(int).values)

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_feats
        g.edata['label'] = labels

        graphs.append(g)

    return graphs

# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate_edge_classifier(graphs, model, device='cpu'):
    """
    Evaluates the edge classifier model on a test set.

    Args:
        graphs (List[dgl.DGLGraph]): List of DGL graphs for evaluation.
        model (nn.Module): EdgeClassifier model.
        device (str): Device to run the evaluation on (CPU or CUDA).

    Returns:
        Tuple[float, float, float]: Returns (avg_loss, accuracy, f1_score).
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    # Initialize Cross-Entropy Loss (assuming no class weights during testing)
    loss_fn = nn.CrossEntropyLoss()

    for g in graphs:
        g = g.to(device)
        node_feats = g.ndata['feat'].float()
        edge_feats = g.edata['feat'].float()
        labels = g.edata['label'].to(device)

        with torch.no_grad():
            logits = model(g, node_feats, edge_feats)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * g.number_of_edges()

            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += g.number_of_edges()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Calculate F1 score for the positive class
    if len(np.unique(all_labels)) == 1:
        # Handle cases where only one class is present
        f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0) if 1 in all_labels else 0.0
    else:
        f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Test F1-Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nSample of Predictions vs Ground Truth:")
    for i in range(min(5, len(all_preds))):
        print(f"Prediction: {all_preds[i]}, Ground Truth: {all_labels[i]}")

    # Additional Metrics for Label '1'
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    ground_truth_ones = np.sum(all_labels_np == 1)
    true_positives = np.sum((all_labels_np == 1) & (all_preds_np == 1))
    false_positives = np.sum((all_labels_np != 1) & (all_preds_np == 1))
    predicted_ones = np.sum(all_preds_np == 1)

    print("\n--- Detailed Metrics for Label '1' ---")
    print(f"Total number of label '1' in ground truth: {ground_truth_ones}")
    print(f"Number of correctly predicted label '1's (True Positives): {true_positives}")
    print(f"Number of incorrectly predicted label '1's (False Positives): {false_positives}")
    print(f"Total number of label '1's predicted: {predicted_ones}")

    return avg_loss, accuracy, f1

# -----------------------------
# Main Function
# -----------------------------

def main(args):
    """
    Main function to execute the testing pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    try:
        nodes_df, edges_df = load_csv_data(args.test_data_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    test_graphs = construct_dgl_graphs(nodes_df, edges_df)

    if len(test_graphs) == 0:
        print("No graphs found in the test dataset. Exiting.")
        sys.exit(1)

    # Initialize model parameters based on the first test graph
    sample_graph = test_graphs[0]
    in_feats = sample_graph.ndata['feat'].shape[1]
    edge_feat_size = sample_graph.edata['feat'].shape[1]
    num_classes = int(sample_graph.edata['label'].max().item() + 1)

    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout

    model = EdgeClassifier(
        in_feats=in_feats,
        hidden_size=hidden_size,
        num_classes=num_classes,
        edge_feat_size=edge_feat_size,
        num_layers=num_layers,
        dropout=dropout
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Move model to device
    model.to(device)

    # Load the best model
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded the best model from '{args.model_path}'.")
    else:
        print(f"Model file not found at '{args.model_path}'. Exiting.")
        sys.exit(1)

    # Evaluate on test set
    import time
    start_time = time.time()
    print("\n--- Test Set Evaluation ---")
    evaluate_edge_classifier(test_graphs, model, device=device)
    end_time = time.time()
    print(f"Total time required {end_time - start_time}s")

# -----------------------------
# Argument Parser
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Edge Classifier on Test Data")
    parser.add_argument('--test_data_dir', type=str, required=True, help='Directory containing test data and meta.yaml')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model (default: best_model.pth)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size (should match training)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GatedGCNConv layers (should match training)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (should match training)')

    args = parser.parse_args()
    main(args)
