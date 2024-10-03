import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import numpy as np

# -----------------------------
# Custom Loss Function Definition
# -----------------------------

class WeightedLoss(nn.Module):
    """
    Custom loss function that applies different weights to false negatives and false positives.

    Args:
        pos_weight (float): Weight for the positive class.
        neg_weight (float): Weight for the negative class.
    """
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.bce_loss(logits.squeeze(), labels.float())
        weights = torch.where(labels == 1, self.pos_weight, self.neg_weight)
        loss = loss * weights
        return loss.mean()

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
# Model Definition
# -----------------------------

class EdgeClassifier(nn.Module):
    """
    Graph Neural Network model for edge classification using GATConv (Graph Attention Network).

    Args:
        in_feats (int): Size of input node features.
        hidden_size (int): Size of hidden layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of classes for edge classification.
        num_layers (int): Number of GATConv layers.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, in_feats, hidden_size, num_heads, num_classes, num_layers=3, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATConv(
                in_feats,
                hidden_size,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
                allow_zero_in_degree=True
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_size * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_size * num_heads,
                    hidden_size,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_size * num_heads))

        # Output layer
        self.convs.append(
            GATConv(
                hidden_size * num_heads,
                hidden_size,
                num_heads=1,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=None,
                allow_zero_in_degree=True
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_size))

        # Edge classification MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Output a single logit for binary classification
        )

    def forward(self, g, node_feats, edge_feats):
        h = node_feats

        for conv, norm in zip(self.convs, self.norms):
            h = conv(g, h)
            if h.dim() == 3:
                # If multi-head outputs, concatenate the heads
                h = h.flatten(1)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        src_h = h[g.edges()[0]]
        dst_h = h[g.edges()[1]]
        edge_repr = torch.cat([src_h, dst_h, edge_feats], dim=1)  # [E, 2 * hidden_size + edge_feat_size]
        logits = self.edge_mlp(edge_repr)
        return logits  # Return raw logits

# -----------------------------
# Training Function
# -----------------------------

def train_edge_classifier(train_graphs, val_graphs, model, epochs=100, lr=0.001, device='cpu', writer=None, early_stopping_patience=20):
    """
    Trains the edge classifier model.

    Args:
        train_graphs (List[dgl.DGLGraph]): List of DGL graphs for training.
        val_graphs (List[dgl.DGLGraph]): List of DGL graphs for validation.
        model (nn.Module): EdgeClassifier model.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to run the training on (CPU or CUDA).
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Scheduler to monitor validation recall (higher is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Compute class weights to emphasize the positive class
    all_train_labels = []
    for g in train_graphs:
        all_train_labels.extend(g.edata['label'].cpu().numpy())
    classes = np.unique(all_train_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=all_train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    # Set higher weight for positive class to penalize false negatives more
    pos_weight = class_weights[1]
    neg_weight = class_weights[0] * 0.5  # Reduce weight for negative class
    loss_fn = WeightedLoss(pos_weight=pos_weight, neg_weight=neg_weight)

    best_val_recall = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        total = 0

        for g in train_graphs:
            g = g.to(device)
            node_feats = g.ndata['feat'].float()
            edge_feats = g.edata['feat'].float()
            labels = g.edata['label'].to(device)

            optimizer.zero_grad()
            logits = model(g, node_feats, edge_feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * g.number_of_edges()
            total += g.number_of_edges()

        avg_loss = epoch_loss / total

        if writer:
            writer.add_scalar('Train/Loss', avg_loss, epoch)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate_edge_classifier(val_graphs, model, device=device, loss_fn=loss_fn, return_metrics=True)
        if val_metrics:
            val_loss, val_recall = val_metrics
        else:
            val_loss, val_recall = None, None

        scheduler.step(val_recall)

        if writer:
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Recall', val_recall, epoch)

        print(f"Validation - Loss: {val_loss:.4f}, Recall: {val_recall:.4f}")

        # Check for improvement based on validation recall
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Validation recall improved to {val_recall:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation recall for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("Training complete.")

# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate_edge_classifier(graphs, model, device='cpu', loss_fn=None, return_metrics=False, threshold=0.5):
    """
    Evaluates the edge classifier model on validation and test sets.

    Args:
        graphs (List[dgl.DGLGraph]): List of DGL graphs for evaluation.
        model (nn.Module): EdgeClassifier model.
        device (str): Device to run the evaluation on (CPU or CUDA).
        loss_fn (nn.Module, optional): Loss function to compute loss. If None, loss is not calculated.
        return_metrics (bool): If True, returns average loss and recall.

    Returns:
        Tuple[float, float] or None: Returns (avg_loss, recall) if return_metrics is True.
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_samples = 0
    all_labels = []
    all_probs = []

    for g in graphs:
        g = g.to(device)
        node_feats = g.ndata['feat'].float()
        edge_feats = g.edata['feat'].float()
        labels = g.edata['label'].to(device)

        with torch.no_grad():
            logits = model(g, node_feats, edge_feats)
            if loss_fn:
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * g.number_of_edges()

            probs = torch.sigmoid(logits.squeeze())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_samples += g.number_of_edges()

    avg_loss = total_loss / total_samples if loss_fn else None

    # Apply threshold to probabilities
    all_preds = (np.array(all_probs) >= threshold).astype(int)

    # Calculate recall for the positive class
    recall = recall_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)

    if loss_fn:
        print(f"Loss: {avg_loss:.4f}, Recall: {recall:.4f}")
    else:
        print(f"Recall: {recall:.4f}")

    # Additional Metrics
    precision = precision_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    print(f"Precision: {precision:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Additional Metrics for Label '1'
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    ground_truth_ones = np.sum(all_labels_np == 1)
    true_positives = np.sum((all_labels_np == 1) & (all_preds_np == 1))
    false_negatives = np.sum((all_labels_np == 1) & (all_preds_np == 0))
    predicted_ones = np.sum(all_preds_np == 1)

    print("\n--- Detailed Metrics for Label '1' ---")
    print(f"Total number of label '1' in ground truth: {ground_truth_ones}")
    print(f"Number of correctly predicted label '1's (True Positives): {true_positives}")
    print(f"Number of missed label '1's (False Negatives): {false_negatives}")
    print(f"Total number of label '1's predicted: {predicted_ones}")

    if return_metrics:
        return avg_loss, recall
    return None

# -----------------------------
# Main Function
# -----------------------------

def main(args):
    """
    Main function to execute the training and evaluation pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    writer = SummaryWriter(log_dir=args.log_dir) if args.use_tensorboard else None

    try:
        nodes_df, edges_df = load_csv_data(args.data_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    graphs = construct_dgl_graphs(nodes_df, edges_df)

    if len(graphs) == 0:
        print("No graphs found. Exiting.")
        sys.exit(1)

    sample_graph = graphs[0]
    in_feats = sample_graph.ndata['feat'].shape[1]
    edge_feat_size = sample_graph.edata['feat'].shape[1]
    num_classes = 2  # Binary classification

    hidden_size = args.hidden_size
    num_heads = args.num_heads
    num_layers = args.num_layers
    dropout = args.dropout
    model = EdgeClassifier(
        in_feats=in_feats,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Move model to device
    model.to(device)

    # Split into Train and Validation only
    num_graphs = len(graphs)
    train_size = int(args.train_ratio * num_graphs)
    val_size = num_graphs - train_size  # Remaining for validation

    if train_size == 0 or val_size == 0:
        print("Insufficient data for the specified train/val split ratios.")
        sys.exit(1)

    train_graphs, val_graphs = train_test_split(
        graphs, train_size=train_size, random_state=42, shuffle=True
    )

    print(f"Total graphs: {num_graphs}")
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")

    # Train the model
    train_edge_classifier(
        train_graphs,
        val_graphs,
        model,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        writer=writer,
        early_stopping_patience=args.early_stopping_patience
    )

    # Save the best model
    if os.path.exists('best_model.pth'):
        print("Best model saved as 'best_model.pth'. You can use this model for testing in a separate script.")
    else:
        print("Best model not found. Ensure training was successful and the model was saved.")

    if writer:
        writer.close()

# -----------------------------
# Argument Parser
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Edge Classification with GATConv for TSP Optimal Edges")
    parser.add_argument('--data_dir', type=str, default='./generated-data', help='Directory containing data and meta.yaml')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GATConv layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')

    args = parser.parse_args()
    main(args)
