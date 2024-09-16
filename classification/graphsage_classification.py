import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import sys
import numpy as np


def load_csv_data(data_directory):
    """
    Loads node and edge data from CSV files specified in a metadata YAML file.

    @param data_directory Directory containing data and metadata files.
    @return Tuple of DataFrames containing node and edge data.
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

    @param nodes_df DataFrame containing node data.
    @param edges_df DataFrame containing edge data.
    @return List of DGL graphs.
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


class EdgeClassifier(nn.Module):
    """
    Graph Neural Network model for edge classification using GraphSAGE.

    @param in_feats Size of input node features.
    @param hidden_size Size of hidden layers.
    @param num_classes Number of classes for edge classification.
    @param edge_feat_size Size of edge features.
    @param num_layers Number of SAGEConv layers.
    @param dropout Dropout rate for regularization.
    """
    def __init__(self, in_feats, hidden_size, num_classes, edge_feat_size=1, num_layers=3, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Input layer
        self.convs.append(SAGEConv(in_feats, hidden_size, 'mean'))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size, 'mean'))
        # Output layer
        self.convs.append(SAGEConv(hidden_size, hidden_size, 'mean'))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + edge_feat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, g, node_feats, edge_feats):
        """
        Forward pass of the model.

        @param g DGL graph.
        @param node_feats Node features.
        @param edge_feats Edge features.
        @return Logits for edge classification.
        """
        h = node_feats
        for conv in self.convs:
            h = conv(g, h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        with g.local_scope():
            g.ndata['h'] = h
            src_h = g.ndata['h'][g.edges()[0]]
            dst_h = g.ndata['h'][g.edges()[1]]
            edge_repr = torch.cat([src_h, dst_h, edge_feats], dim=1)
            logits = self.edge_mlp(edge_repr)
        return logits


def train_edge_classifier(train_graphs, model, epochs=100, lr=0.001, device='cpu', writer=None, early_stopping_patience=20):
    """
    Trains the edge classifier model.

    @param train_graphs List of DGL graphs for training.
    @param model EdgeClassifier model.
    @param epochs Number of training epochs.
    @param lr Learning rate.
    @param device Device to run the training on (CPU or CUDA).
    @param writer TensorBoard writer for logging.
    @param early_stopping_patience Number of epochs with no improvement after which training will be stopped.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total = 0

        # Batch graphs
        batched_graph = dgl.batch(train_graphs)
        batched_graph = batched_graph.to(device)
        node_feats = batched_graph.ndata['feat'].to(device).float()
        edge_feats = batched_graph.edata['feat'].to(device).float()
        labels = batched_graph.edata['label'].to(device)

        optimizer.zero_grad()
        logits = model(batched_graph, node_feats, edge_feats)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batched_graph.number_of_edges()

        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == labels).sum().item()
        epoch_correct += correct
        total += batched_graph.number_of_edges()

        avg_loss = epoch_loss / total
        avg_acc = epoch_correct / total

        if writer:
            writer.add_scalar('Train/Loss', avg_loss, epoch)
            writer.add_scalar('Train/Accuracy', avg_acc, epoch)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc:.4f}")

        # Implement early stopping based on validation loss
        if epoch % 1 == 0:
            val_loss, val_acc = evaluate_edge_classifier(train_graphs, model, device=device, loss_fn=loss_fn, return_metrics=True)
            scheduler.step(val_loss)
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


def evaluate_edge_classifier(graphs, model, device='cpu', loss_fn=None, return_metrics=False):
    """
    Evaluates the edge classifier model on validation and test sets.

    @param graphs List of DGL graphs for evaluation.
    @param model EdgeClassifier model.
    @param device Device to run the evaluation on (CPU or CUDA).
    @param loss_fn Loss function to compute loss. If None, loss is not calculated.
    @param return_metrics If True, returns average loss and accuracy.
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
        labels = batched_graph.edata['label'].to(device)
        
        logits = model(batched_graph, node_feats, edge_feats)
        if loss_fn:
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * batched_graph.number_of_edges()
        
        _, predicted = torch.max(logits, dim=1)
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

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
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
    # print("\nSample of Predictions vs Ground Truth:")
    # for i in range(min(5, len(all_preds))):
    #     print(f"Prediction: {all_preds[i]}, Ground Truth: {all_labels[i]}")


def main(args):
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
    
    loss_fn = nn.CrossEntropyLoss()
    
    train_edge_classifier(
        train_graphs, 
        model, 
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
    evaluate_edge_classifier(val_graphs, model, device=device, loss_fn=loss_fn)

    print("\n--- Test Set Evaluation ---")
    evaluate_edge_classifier(test_graphs, model, device=device, loss_fn=loss_fn)

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Edge Classification with GNN")
    parser.add_argument('--data_dir', type=str, default='./generated-data', help='Directory containing data and meta.yaml')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of data for validation')
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--early_stopping_patience', type=int, default=100, help='Patience for early stopping')

    args = parser.parse_args()
    main(args)
