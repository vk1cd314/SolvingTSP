import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import numpy as np


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
        labels = torch.tensor(edges['label'].astype(int).values).float()  # Binary labels

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_feats
        g.edata['label'] = labels

        graphs.append(g)

    return graphs


class GraphAttentionLayer(nn.Module):
    """
    Custom Multi-Head Graph Attention Layer that incorporates edge features.

    Args:
        in_feats (int): Size of input node features.
        edge_feats (int): Size of edge features.
        out_feats (int): Size of output node features per head.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        alpha (float): Negative slope for LeakyReLU.
        concat (bool): Whether to concatenate the attention heads' output.
    """
    def __init__(self, in_feats, edge_feats, out_feats, num_heads=4, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_feats = in_feats
        self.edge_feats = edge_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.concat = concat

        # Linear layers for node and edge features
        self.W = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.W_e = nn.Linear(edge_feats, out_feats * num_heads, bias=False)

        # Attention mechanism parameters for each head
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, out_feats, 1))
        self.a_dst = nn.Parameter(torch.FloatTensor(num_heads, out_feats, 1))
        self.a_edge = nn.Parameter(torch.FloatTensor(num_heads, out_feats, 1))

        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_e.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_edge.data, gain=1.414)

    def edge_attention(self, edges):
        # Compute attention scores
        src = edges.src['Wh']  # (E, num_heads, out_feats)
        dst = edges.dst['Wh']  # (E, num_heads, out_feats)
        e = edges.data['We']    # (E, num_heads, out_feats)

        # Compute attention score using einsum for efficiency
        # (E, num_heads, out_feats) * (num_heads, out_feats) -> (E, num_heads)
        score = torch.einsum('ehf,hf->eh', src, self.a_src.squeeze(-1)) + \
                torch.einsum('ehf,hf->eh', dst, self.a_dst.squeeze(-1)) + \
                torch.einsum('ehf,hf->eh', e, self.a_edge.squeeze(-1))  # (E, num_heads)

        score = F.leaky_relu(score, self.alpha)  # (E, num_heads)

        return {'score': score.contiguous()}  # Ensure contiguity

    def message_func(self, edges):
        # Multiply the transformed node features by the attention scores
        # edges.src['Wh']: (E, num_heads, out_feats)
        # edges.data['a']: (E, num_heads, 1)
        return {'Wh': edges.src['Wh'] * edges.data['a']}  # (E, num_heads, out_feats)

    def forward(self, g, h, edge_features):
        """
        Forward pass of the Graph Attention Layer.

        Args:
            g (dgl.DGLGraph): The input graph.
            h (torch.Tensor): Node features of shape (N, in_feats).
            edge_features (torch.Tensor): Edge features of shape (E, edge_feats).

        Returns:
            torch.Tensor: Updated node features of shape (N, num_heads * out_feats) if concat=True,
                          else (N, out_feats).
        """
        Wh = self.W(h).view(-1, self.num_heads, self.out_feats)          # (N, num_heads, out_feats)
        We = self.W_e(edge_features).view(-1, self.num_heads, self.out_feats)  # (E, num_heads, out_feats)

        g = g.local_var()
        g.ndata['Wh'] = Wh
        g.edata['We'] = We

        # Compute attention scores
        g.apply_edges(self.edge_attention)  # Stores 'score' in edges

        # Perform softmax over the attention scores for each head
        scores = g.edata.pop('score')       # (E, num_heads)
        scores = scores.contiguous().unsqueeze(-1)  # (E, num_heads, 1)
        g.edata['a'] = edge_softmax(g, scores.squeeze(-1)).unsqueeze(-1)  # (E, num_heads, 1)

        # Apply dropout to attention scores
        g.edata['a'] = self.dropout(g.edata['a'])

        # Send messages with attention scores
        g.update_all(self.message_func, fn.sum(msg='Wh', out='h_new'))

        h_new = g.ndata['h_new']  # (N, num_heads, out_feats)

        if self.concat:
            # Concatenate the heads
            h_new = h_new.view(-1, self.num_heads * self.out_feats)  # (N, num_heads * out_feats)
            return F.elu(h_new)
        else:
            # Average the heads
            h_new = h_new.mean(dim=1)  # (N, out_feats)
            return h_new


class EdgeClassifier(nn.Module):
    """
    Graph Neural Network model for edge classification using custom GraphAttentionLayer.

    Args:
        in_feats (int): Size of input node features.
        hidden_size (int): Size of hidden layers.
        num_classes (int): Number of classes for edge classification (set to 1 for binary classification).
        edge_feat_size (int): Size of edge features.
        num_layers (int): Number of GraphAttentionLayer layers.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, in_feats, hidden_size, num_classes, edge_feat_size=1, num_layers=3, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(
            GraphAttentionLayer(
                in_feats=in_feats,
                edge_feats=edge_feat_size,
                out_feats=hidden_size,
                num_heads=4,           # You can adjust the number of heads
                dropout=dropout,
                alpha=0.2,
                concat=True
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GraphAttentionLayer(
                    in_feats=hidden_size * 4,  # Because concat=True and num_heads=4
                    edge_feats=edge_feat_size,
                    out_feats=hidden_size,
                    num_heads=4,
                    dropout=dropout,
                    alpha=0.2,
                    concat=True
                )
            )

        # Output layer
        self.convs.append(
            GraphAttentionLayer(
                in_feats=hidden_size * 4,
                edge_feats=edge_feat_size,
                out_feats=hidden_size,
                num_heads=1,          # Single head for the final layer
                dropout=dropout,
                alpha=0.2,
                concat=False          # No concatenation, output shape (N, hidden_size)
            )
        )

        # For binary classification, output a single logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + edge_feat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Single output for binary classification
        )

    def forward(self, g, node_feats, edge_feats):
        """
        Forward pass of the model.

        Args:
            g (dgl.DGLGraph): The input graph.
            node_feats (torch.Tensor): Node features of shape (N, in_feats).
            edge_feats (torch.Tensor): Edge features of shape (E, edge_feat_size).

        Returns:
            torch.Tensor: Logits for edge classification of shape (E, 1).
        """
        h = node_feats
        e = edge_feats

        for conv in self.convs:
            h = conv(g, h, e)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # After all GraphAttentionLayers, h has shape (N, hidden_size)
        # To classify edges, concatenate source and destination node representations along with edge features
        src, dst = g.edges()
        src_h = h[src]  # (E, hidden_size)
        dst_h = h[dst]  # (E, hidden_size)
        edge_repr = torch.cat([src_h, dst_h, e], dim=1)  # [E, 2 * hidden_size + edge_feat_size]
        logits = self.edge_mlp(edge_repr)  # (E, 1)
        return logits.squeeze(-1)  # (E,)


def train_edge_classifier(
    train_graphs, 
    val_graphs, 
    model, 
    class_weights,          # Added parameter
    epochs=100, 
    lr=0.001, 
    device='cpu', 
    writer=None, 
    early_stopping_patience=20
):
    """
    Trains the edge classifier model.

    Args:
        train_graphs (List[dgl.DGLGraph]): List of DGL graphs for training.
        val_graphs (List[dgl.DGLGraph]): List of DGL graphs for validation.
        model (nn.Module): EdgeClassifier model.
        class_weights (torch.Tensor): Tensor of class weights.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to run the training on (CPU or CUDA).
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # **Initialize the loss function with class weights**
    # For BCEWithLogitsLoss, pos_weight should be set to (weight for class 1) / (weight for class 0)
    pos_weight = class_weights[1] / class_weights[0]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total = 0

        # Batch training graphs
        batched_graph = dgl.batch(train_graphs)
        batched_graph = batched_graph.to(device)
        node_feats = batched_graph.ndata['feat'].to(device).float()
        edge_feats = batched_graph.edata['feat'].to(device).float()
        labels = batched_graph.edata['label'].to(device)  # (E,)

        optimizer.zero_grad()
        logits = model(batched_graph, node_feats, edge_feats)  # (E,)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batched_graph.number_of_edges()

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        # Define a threshold; to ensure all true 1s are predicted, set threshold low enough
        # This threshold can be adjusted based on validation set
        threshold = 0.5  # Initial threshold; consider adjusting
        predicted = (probs >= threshold).float()
        correct = (predicted == labels).sum().item()
        epoch_correct += correct
        total += batched_graph.number_of_edges()

        avg_loss = epoch_loss / total
        avg_acc = epoch_correct / total

        if writer:
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            writer.add_scalar('Train/Accuracy', avg_acc, epoch)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc:.4f}")

        # Evaluate on validation set for early stopping
        val_loss, val_acc = evaluate_edge_classifier(val_graphs, model, device=device, loss_fn=loss_fn, return_metrics=True, threshold=threshold)
        scheduler.step(val_loss)

        if writer:
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("Training complete.")


def evaluate_edge_classifier(graphs, model, device='cpu', loss_fn=None, return_metrics=False, threshold=0.5):
    """
    Evaluates the edge classifier model on validation and test sets.

    Args:
        graphs (List[dgl.DGLGraph]): List of DGL graphs for evaluation.
        model (nn.Module): EdgeClassifier model.
        device (str): Device to run the evaluation on (CPU or CUDA).
        loss_fn (nn.Module, optional): Loss function to compute loss. If None, loss is not calculated.
        return_metrics (bool): If True, returns average loss and accuracy.
        threshold (float): Threshold for classifying edges as class 1.

    Returns:
        Tuple[float, float] or None: Returns (avg_loss, accuracy) if return_metrics is True.
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        batched_graph = dgl.batch(graphs)
        batched_graph = batched_graph.to(device)
        node_feats = batched_graph.ndata['feat'].to(device).float()
        edge_feats = batched_graph.edata['feat'].to(device).float()
        labels = batched_graph.edata['label'].to(device)  # (E,)

        logits = model(batched_graph, node_feats, edge_feats)  # (E,)
        if loss_fn:
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * batched_graph.number_of_edges()

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        predicted = (probs >= threshold).float()
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += batched_graph.number_of_edges()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / total_samples if loss_fn else None
    accuracy = total_correct / total_samples

    if return_metrics:
        return avg_loss, accuracy

    if loss_fn:
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    else:
        print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    print("\nSample of Predictions vs Ground Truth:")
    for i in range(min(5, len(all_preds))):
        print(f"Prediction: {int(all_preds[i])}, Ground Truth: {int(all_labels[i])}")

    # **Begin: Additional Metrics for Label '1'**
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # 1. Total number of label '1' in ground truth
    ground_truth_ones = np.sum(all_labels_np == 1)

    # 2. Number of correctly predicted label '1's (True Positives)
    true_positives = np.sum((all_labels_np == 1) & (all_preds_np == 1))

    # 3. Number of incorrectly predicted label '1's (False Positives)
    false_positives = np.sum((all_labels_np != 1) & (all_preds_np == 1))

    # 4. Total number of label '1's predicted (True Positives + False Positives)
    predicted_ones = np.sum(all_preds_np == 1)

    print("\n--- Detailed Metrics for Label '1' ---")
    print(f"Total number of label '1' in ground truth: {ground_truth_ones}")
    print(f"Number of correctly predicted label '1's (True Positives): {true_positives}")
    print(f"Number of incorrectly predicted label '1's (False Positives): {false_positives}")
    print(f"Total number of label '1's predicted: {predicted_ones}")
    # **End: Additional Metrics for Label '1'**


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
    num_classes = 1  # Binary classification

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

    # **Define Fixed Class Weights Here**
    # Class 0 weight: 0.01, Class 1 weight: 0.99
    class_weights = torch.tensor([0.1, 0.9], dtype=torch.float)
    # For BCEWithLogitsLoss, pos_weight is (weight for class 1) / (weight for class 0)
    pos_weight = class_weights[1] / class_weights[0]
    pos_weight = pos_weight.to(device)
    print(f"Using fixed class weights: Class 0: 0.1, Class 1: 0.9")
    print(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight.item()}")

    # Split data into train, validation, and test sets
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

    # **Pass class_weights to the training function**
    train_edge_classifier(
        train_graphs,
        val_graphs,
        model,
        class_weights=class_weights,   # Added this argument
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        writer=writer,
        early_stopping_patience=args.early_stopping_patience
    )

    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded the best model from checkpoint.")

    print("\n--- Validation Set Evaluation ---")
    evaluate_edge_classifier(val_graphs, model, device=device, loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight))

    print("\n--- Test Set Evaluation ---")
    evaluate_edge_classifier(test_graphs, model, device=device, loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight))

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Edge Classification with Enhanced GraphAttentionLayer")
    parser.add_argument('--data_dir', type=str, default='./generated-data', help='Directory containing data and meta.yaml')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GraphAttentionLayer layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of data for validation')
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Patience for early stopping')

    args = parser.parse_args()
    main(args)
