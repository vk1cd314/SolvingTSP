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

def load_csv_data(data_directory):
    """
    Loads node and edge data from CSV files specified in a metadata YAML file.

    @param data_directory Directory containing data and metadata files.
    @return Tuple of DataFrames containing node and edge data.
    """
    meta_path = os.path.join(data_directory, 'meta.yaml')
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    node_file = os.path.join(data_directory, meta['node_data'][0]['file_name'])
    edge_file = os.path.join(data_directory, meta['edge_data'][0]['file_name'])
    
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
    """
    def __init__(self, in_feats, hidden_size, num_classes, edge_feat_size=1):
        super(EdgeClassifier, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, 'gcn')
        self.conv2 = SAGEConv(hidden_size, hidden_size, 'gcn')
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + edge_feat_size, hidden_size),
            nn.ReLU(),
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
        h = self.conv1(g, node_feats)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        
        with g.local_scope():
            g.ndata['h'] = h
            src_h = g.ndata['h'][g.edges()[0]]
            dst_h = g.ndata['h'][g.edges()[1]]
            edge_repr = torch.cat([src_h, dst_h, edge_feats], dim=1)
            logits = self.edge_mlp(edge_repr)
        return logits

def train_edge_classifier(train_graphs, model, class_weights, epochs=100, lr=0.001, device='cpu'):
    """
    Trains the edge classifier model.

    @param train_graphs List of DGL graphs for training.
    @param model EdgeClassifier model.
    @param class_weights Tensor of class weights.
    @param epochs Number of training epochs.
    @param lr Learning rate.
    @param device Device to run the training on (CPU or CUDA).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        total = 0
        
        for g in train_graphs:
            g = g.to(device)
            node_feats = g.ndata['feat'].to(device).float()
            edge_feats = g.edata['feat'].to(device).float()
            labels = g.edata['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(g, node_feats, edge_feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * g.number_of_edges()
            
            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels).sum().item()
            epoch_acc += correct
            total += g.number_of_edges()
        
        avg_loss = epoch_loss / total
        avg_acc = epoch_acc / total
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc:.4f}")
    
    print("Training complete.")

def evaluate_edge_classifier(graphs, model, device='cpu'):
    """
    Evaluates the edge classifier model on validation and test sets.

    @param graphs List of DGL graphs for evaluation.
    @param model EdgeClassifier model.
    @param device Device to run the evaluation on (CPU or CUDA).
    """
    model.eval()
    model.to(device)
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    
    # For evaluation, we typically use uniform loss without class weights
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            node_feats = g.ndata['feat'].to(device).float()
            edge_feats = g.edata['feat'].to(device).float()
            labels = g.edata['label'].to(device)
            
            logits = model(g, node_feats, edge_feats)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * g.number_of_edges()
            
            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += g.number_of_edges()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
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
    print("\nSample of Predictions vs Ground Truth:")
    for i in range(min(5, len(all_preds))):
        print(f"Prediction: {all_preds[i]}, Ground Truth: {all_labels[i]}")

if __name__ == "__main__":
    # Load data
    nodes_df, edges_df = load_csv_data('./generated-data')
    graphs = construct_dgl_graphs(nodes_df, edges_df)
    
    # Prepare model parameters
    sample_graph = graphs[0]
    in_feats = sample_graph.ndata['feat'].shape[1]
    edge_feat_size = sample_graph.edata['feat'].shape[1]
    num_classes = sample_graph.edata['label'].max().item() + 1
    
    hidden_size = 64
    model = EdgeClassifier(in_feats, hidden_size, num_classes, edge_feat_size=edge_feat_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Split data into train, validation, and test sets
    num_graphs = len(graphs)
    train_size = int(0.6 * num_graphs)
    val_size = int(0.2 * num_graphs)
    test_size = num_graphs - train_size - val_size  # Ensures all graphs are used
    
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
    
    # **Set Fixed Class Weights Here**
    # Class 0 weight: 0.01, Class 1 weight: 0.99
    class_weights = torch.tensor([0.01, 0.99], dtype=torch.float)
    print(f"Using fixed class weights: {class_weights}")
    
    # Train the model with fixed class weights
    train_edge_classifier(train_graphs, model, class_weights, epochs=100, lr=0.001, device=device)
    
    # Evaluate on Validation Set
    print("\n--- Validation Set Evaluation ---")
    evaluate_edge_classifier(val_graphs, model, device=device)
    
    # Evaluate on Test Set
    print("\n--- Test Set Evaluation ---")
    evaluate_edge_classifier(test_graphs, model, device=device)
