import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GatedGCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight  # Added import
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import numpy as np

# -----------------------------
# Custom Loss Function Definition
# -----------------------------

class CustomLoss(nn.Module):
    """
    Custom loss function that combines Cross-Entropy Loss with a regularization
    term to penalize the number of '1's predicted.
    
    Args:
        class_weights (torch.Tensor): Tensor containing weights for each class.
        lambda_reg (float): Regularization coefficient to control the penalty for '1' predictions.
    """
    def __init__(self, class_weights, lambda_reg=0.01):
        super(CustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.lambda_reg = lambda_reg
    
    def forward(self, logits, labels):
        # Compute Cross-Entropy Loss
        ce = self.cross_entropy(logits, labels)
        
        # Compute the number of '1's predicted
        preds = torch.argmax(logits, dim=1)
        num_ones = (preds == 1).float().sum()
        
        # Regularization term to penalize '1' predictions
        reg = self.lambda_reg * num_ones
        
        # Total Loss
        return ce + reg

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
# Training Function
# -----------------------------

def train_edge_classifier(train_graphs, val_graphs, model, epochs=100, lr=0.001, device='cpu', writer=None, early_stopping_patience=20, lambda_reg=0.01):
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
        lambda_reg (float): Regularization coefficient for minimizing '1' predictions.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Scheduler to monitor F1 score (higher is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # Compute class weights: higher weight for '1's to reward correct identification
    all_train_labels = []
    for g in train_graphs:
        all_train_labels.extend(g.edata['label'].cpu().numpy())
    classes = np.unique(all_train_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=all_train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
    
    # Initialize Custom Loss
    loss_fn = CustomLoss(class_weights=class_weights, lambda_reg=lambda_reg)

    best_val_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total = 0
        epoch_f1 = 0.0

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

            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels).sum().item()
            epoch_correct += correct
            total += g.number_of_edges()

            # Calculate F1 score for this graph
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), pos_label=1, average='binary', zero_division=0)
            epoch_f1 += f1

        avg_loss = epoch_loss / total
        avg_acc = epoch_correct / total
        avg_f1 = epoch_f1 / len(train_graphs)  # Average F1 over all training graphs

        if writer:
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            writer.add_scalar('Train/Accuracy', avg_acc, epoch)
            writer.add_scalar('Train/F1_Score', avg_f1, epoch)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc:.4f}, F1-Score: {avg_f1:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate_edge_classifier(val_graphs, model, device=device, loss_fn=loss_fn, return_metrics=True)
        if val_metrics:
            val_loss, val_acc, val_f1 = val_metrics
        else:
            val_loss, val_acc, val_f1 = None, None, None

        scheduler.step(val_f1)

        if writer:
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)
            writer.add_scalar('Validation/F1_Score', val_f1, epoch)

        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1-Score: {val_f1:.4f}")

        # Check for improvement based on F1-Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Validation F1 improved to {val_f1:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation F1 for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("Training complete.")

# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate_edge_classifier(graphs, model, device='cpu', loss_fn=None, return_metrics=False):
    """
    Evaluates the edge classifier model on validation and test sets.

    Args:
        graphs (List[dgl.DGLGraph]): List of DGL graphs for evaluation.
        model (nn.Module): EdgeClassifier model.
        device (str): Device to run the evaluation on (CPU or CUDA).
        loss_fn (nn.Module, optional): Loss function to compute loss. If None, loss is not calculated.
        return_metrics (bool): If True, returns average loss, accuracy, and F1-score.

    Returns:
        Tuple[float, float, float] or None: Returns (avg_loss, accuracy, f1_score) if return_metrics is True.
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

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

            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += g.number_of_edges()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / total_samples if loss_fn else None
    accuracy = total_correct / total_samples

    # Calculate F1 score for the positive class
    if len(np.unique(all_labels)) == 1:
        # Handle cases where only one class is present
        f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0) if 1 in all_labels else 0.0
    else:
        f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)

    if loss_fn:
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    else:
        print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

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
    if return_metrics:
        return avg_loss, accuracy, f1
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

    # # Add the computational graph to TensorBoard
    # if writer:
    #     # Select a sample graph and move it to the device
    #     sample_g = graphs[0].to(device)
    #     node_feats = sample_g.ndata['feat'].float()
    #     edge_feats = sample_g.edata['feat'].float()
    #
    #     # Add graph to TensorBoard
    #     try:
    #         writer.add_graph(model, (sample_g, node_feats, edge_feats))
    #         print("Model graph added to TensorBoard.")
    #     except Exception as e:
    #         print(f"Failed to add graph to TensorBoard: {e}")

    num_graphs = len(graphs)
    train_size = int(args.train_ratio * num_graphs)
    val_size = int(args.val_ratio * num_graphs)
    test_size = num_graphs - train_size - val_size  # Ensures all graphs are used

    if train_size == 0 or val_size == 0 or test_size == 0:
        print("Insufficient data for the specified train/val/test split ratios.")
        sys.exit(1)

    train_graphs, temp_graphs = train_test_split(
        graphs, train_size=train_size, random_state=42, shuffle=True
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"Total graphs: {num_graphs}")
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Testing graphs: {len(test_graphs)}")

    # Train the model
    train_edge_classifier(
        train_graphs,
        val_graphs,
        model,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        writer=writer,
        early_stopping_patience=args.early_stopping_patience,
        lambda_reg=args.lambda_reg  # Pass the regularization parameter
    )

    # Load the best model
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("Loaded the best model from checkpoint.")

    # Evaluate on validation set
    print("\n--- Validation Set Evaluation ---")
    val_metrics = evaluate_edge_classifier(
        val_graphs, model, device=device, loss_fn=None, return_metrics=True  # Use default loss for final evaluation
    )
    if val_metrics:
        val_loss, val_acc, val_f1 = val_metrics
        if writer:
            if val_loss is not None:
                writer.add_scalar('Final Evaluation/Validation Loss', val_loss, 0)
            writer.add_scalar('Final Evaluation/Validation Accuracy', val_acc, 0)
            writer.add_scalar('Final Evaluation/Validation F1_Score', val_f1, 0)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_metrics = evaluate_edge_classifier(
        test_graphs, model, device=device, loss_fn=None, return_metrics=True  # Use default loss for final evaluation
    )
    if test_metrics:
        test_loss, test_acc, test_f1 = test_metrics
        if writer:
            if test_loss is not None:
                writer.add_scalar('Final Evaluation/Test Loss', test_loss, 0)
            writer.add_scalar('Final Evaluation/Test Accuracy', test_acc, 0)
            writer.add_scalar('Final Evaluation/Test F1_Score', test_f1, 0)

    if writer:
        writer.close()

# -----------------------------
# Argument Parser
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Edge Classification with GatedGCNConv")
    parser.add_argument('--data_dir', type=str, default='./generated-data', help='Directory containing data and meta.yaml')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GatedGCNConv layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of data for validation')
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--early_stopping_patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--lambda_reg', type=float, default=0.5, help='Regularization coefficient for minimizing \'1\' predictions')

    args = parser.parse_args()
    main(args)
