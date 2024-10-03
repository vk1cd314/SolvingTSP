import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
from dgl.dataloading import DataLoader, NeighborSampler, as_edge_prediction_sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import numpy as np
import scipy
from scipy.sparse import csgraph
from scipy.linalg import eigh
# from dgl import enable_cpu_affinity
# enable_cpu_affinity()
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

def normalize_features(nodes_df, edges_df):
    """
    Normalizes node and edge features using z-score normalization.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node data.
        edges_df (pd.DataFrame): DataFrame containing edge data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Normalized node and edge DataFrames.
    """
    # Normalize node features
    node_feat = nodes_df['feat'].astype(float)
    nodes_df['feat'] = (node_feat - node_feat.mean()) / node_feat.std()

    # Normalize edge features
    edge_feat = edges_df['feat'].astype(float)
    edges_df['feat'] = (edge_feat - edge_feat.mean()) / edge_feat.std()

    return nodes_df, edges_df


def add_positional_encodings(g, k=10):
    n = g.number_of_nodes()
    src, dst = g.edges()
    data = np.ones(len(src))
    adj = scipy.sparse.csr_matrix((data, (src.numpy(), dst.numpy())), shape=(n, n))
    # Ensure the adjacency matrix is symmetric
    adj = adj + adj.T
    adj[adj > 1] = 1
    # Compute the Laplacian
    laplacian = csgraph.laplacian(adj, normed=True)
    eigenvals, eigenvecs = eigh(laplacian.toarray())
    pe = torch.from_numpy(eigenvecs[:, 1:k+1]).float()
    if pe.shape[1] < k:
        padding = torch.zeros(n, k - pe.shape[1])
        pe = torch.cat([pe, padding], dim=1)
    g.ndata['feat'] = torch.cat([g.ndata['feat'], pe], dim=1)
    return g



def construct_dgl_graphs(nodes_df, edges_df, use_pos_encodings=True, k=10):
    """
    Constructs a list of DGL graphs from node and edge data.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node data.
        edges_df (pd.DataFrame): DataFrame containing edge data.
        use_pos_encodings (bool): Whether to add positional encodings.
        k (int): Number of eigenvectors for positional encodings.

    Returns:
        List[dgl.DGLGraph]: List of DGL graphs.
    """
    # Normalize features
    nodes_df, edges_df = normalize_features(nodes_df, edges_df)

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

        if use_pos_encodings:
            g = add_positional_encodings(g, k=k)

        graphs.append(g)

    return graphs

# -----------------------------
# Model Definition
# -----------------------------

class EdgeClassifier(nn.Module):
    """
    Graph Neural Network model for edge classification using SAGEConv.

    Args:
        in_feats (int): Size of input node features.
        hidden_size (int): Size of hidden layers.
        num_classes (int): Number of classes for edge classification.
        num_layers (int): Number of SAGEConv layers.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, in_feats, hidden_size, num_classes, num_layers=3, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type='mean'))
        self.norms.append(nn.BatchNorm1d(hidden_size))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type='mean'))
            self.norms.append(nn.BatchNorm1d(hidden_size))

        # Output layer
        self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type='mean'))
        self.norms.append(nn.BatchNorm1d(hidden_size))

        # Edge classification MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Output a single logit for binary classification
        )

    def forward(self, blocks, node_feats, edge_feats, pair_graph):
        h = node_feats
        for layer, norm, block in zip(self.layers, self.norms, blocks):
            h = layer(block, h)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Get the node embeddings for the source and destination nodes in pair_graph
        src, dst = pair_graph.edges()
        src_h = h[src]
        dst_h = h[dst]

        edge_repr = torch.cat([src_h, dst_h, edge_feats], dim=1)
        logits = self.edge_mlp(edge_repr)
        return logits

# -----------------------------
# Training Function
# -----------------------------

def train_edge_classifier(train_loader, val_loader, model, num_layers, epochs=100, lr=0.001, device='cpu', writer=None, early_stopping_patience=20):
    """
    Trains the edge classifier model.

    Args:
        train_loader (iterable): DataLoader for training data.
        val_loader (iterable): DataLoader for validation data.
        model (nn.Module): EdgeClassifier model.
        num_layers (int): Number of GNN layers.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to run the training on (CPU or CUDA).
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Scheduler to monitor validation recall (higher is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print("what is ")
    # Compute class weights to emphasize the positive class
    all_train_labels = []
    batch_no = 0
    for input_nodes, pair_graph, blocks in train_loader:
        all_train_labels.extend(pair_graph.edata['label'].cpu().numpy())
        batch_no += 1
        if batch_no % 1000 == 0:
            print(batch_no)
    classes = np.unique(all_train_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=all_train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
    print("going on?")

    # Set higher weight for positive class to penalize false negatives more
    pos_weight = class_weights[1]
    neg_weight = class_weights[0] * 0.5  # Reduce weight for negative class
    loss_fn = WeightedLoss(pos_weight=pos_weight, neg_weight=neg_weight)

    best_val_recall = 0.0
    epochs_no_improve = 0

    print(f"Is train_loader empty? {len(list(train_loader)) == 0}")
    print("whatt????")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        total = 0

        for input_nodes, pair_graph, blocks in train_loader:
            print("whatt????")
            blocks = [block.to(device) for block in blocks]
            pair_graph = pair_graph.to(device)
            node_feats = blocks[0].srcdata['feat']
            edge_feats = pair_graph.edata['feat']
            labels = pair_graph.edata['label'].to(device)

            optimizer.zero_grad()
            logits = model(blocks, node_feats, edge_feats, pair_graph)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * pair_graph.number_of_edges()
            total += pair_graph.number_of_edges()

        avg_loss = epoch_loss / total

        if writer:
            writer.add_scalar('Train/Loss', avg_loss, epoch)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_loss, val_recall = evaluate_edge_classifier(val_loader, model, num_layers, device=device, loss_fn=loss_fn, return_metrics=True)
        scheduler.step(val_recall)

        if writer:
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Recall', val_recall, epoch)

        print(f"Validation - Loss: {val_loss:.4f}, Recall: {val_recall:.4f}")

        # Check for improvement
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            epochs_no_improve = 0
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

def evaluate_edge_classifier(loader, model, num_layers, device='cpu', loss_fn=None, return_metrics=False, threshold=0.5):
    """
    Evaluates the edge classifier model on validation and test sets.

    Args:
        loader (iterable): DataLoader for evaluation data.
        model (nn.Module): EdgeClassifier model.
        num_layers (int): Number of GNN layers.
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

    with torch.no_grad():
        for input_nodes, pair_graph, blocks in loader:
            blocks = [block.to(device) for block in blocks]
            pair_graph = pair_graph.to(device)
            node_feats = blocks[0].srcdata['feat']
            edge_feats = pair_graph.edata['feat']
            labels = pair_graph.edata['label'].to(device)

            logits = model(blocks, node_feats, edge_feats, pair_graph)
            if loss_fn:
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * pair_graph.number_of_edges()

            probs = torch.sigmoid(logits.squeeze())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_samples += pair_graph.number_of_edges()

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
# Data Loader Creation Function
# -----------------------------

def create_dataloader(graphs, num_layers, batch_size, num_neighbors):
    """
    Creates DataLoaders with neighbor sampling for the graphs.

    Args:
        graphs (List[dgl.DGLGraph]): List of DGL graphs.
        num_layers (int): Number of GNN layers.
        batch_size (int): Batch size for DataLoader.
        num_neighbors (int): Number of neighbors to sample.

    Returns:
        List[DataLoader]: List of DataLoaders for the graphs.
    """
    loaders = []
    
    idx = 0
    for g in graphs:
        print(idx)
        idx += 1
        sampler = as_edge_prediction_sampler(
            NeighborSampler([num_neighbors] * num_layers),
            exclude='self'
        )
        # Use all edges for training/evaluation
        eids = torch.arange(g.number_of_edges())
        # Create a DataLoader
        dataloader = DataLoader(
            g,
            eids,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4
        )
        loaders.append(dataloader)

    return loaders

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

    graphs = construct_dgl_graphs(nodes_df, edges_df, use_pos_encodings=True, k=args.pos_enc_dim)

    if len(graphs) == 0:
        print("No graphs found. Exiting.")
        sys.exit(1)
    
    print("Finished Constructing them graphs")
    sample_graph = graphs[0]
    in_feats = sample_graph.ndata['feat'].shape[1]
    num_classes = 2  # Binary classification

    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    model = EdgeClassifier(
        in_feats=in_feats,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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

    # Create dataloaders with neighbor sampling
    train_loaders = create_dataloader(train_graphs, num_layers, args.batch_size, args.num_neighbors)
    val_loaders = create_dataloader(val_graphs, num_layers, args.batch_size, args.num_neighbors)

    # Combine loaders
    from itertools import chain
    train_loader = chain(*train_loaders)
    val_loader = chain(*val_loaders)
    
    print("Chained")
    # Train the model
    train_edge_classifier(
        train_loader,
        val_loader,
        model,
        num_layers,
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
    parser = argparse.ArgumentParser(description="Graph Edge Classification with Generalization to Larger Graphs")
    parser.add_argument('--data_dir', type=str, default='./generated-data', help='Directory containing data and meta.yaml')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--use_tensorboard', action='store_true', default=False, help='Enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_neighbors', type=int, default=2, help='Number of neighbors to sample')
    parser.add_argument('--pos_enc_dim', type=int, default=10, help='Dimension of positional encodings')

    args = parser.parse_args()
    main(args)
