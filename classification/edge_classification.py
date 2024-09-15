import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
from sklearn.model_selection import train_test_split
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
        
        g = split_edges(g)
        graphs.append(g)
    
    return graphs

def split_edges(g, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Splits edges of a graph into training, validation, and test sets.

    @param g DGL graph.
    @param train_ratio Ratio of edges used for training.
    @param val_ratio Ratio of edges used for validation.
    @param test_ratio Ratio of edges used for testing.
    @return Graph with train, validation, and test masks.
    """
    num_edges = g.number_of_edges()
    all_indices = torch.arange(num_edges)
    labels = g.edata['label'].numpy()
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        raise ValueError("At least two samples are required for each class to perform stratified splitting.")
    
    train_idx, temp_idx = train_test_split(
        all_indices.numpy(),
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=labels
    )
    
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(test_ratio / (test_ratio + val_ratio)),
        random_state=42,
        stratify=temp_labels
    )
    
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    g.edata['train_mask'] = train_mask
    g.edata['val_mask'] = val_mask
    g.edata['test_mask'] = test_mask
    
    return g

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
        self.conv1 = SAGEConv(in_feats, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, hidden_size, 'mean')
        
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

def train_edge_classifier(graphs, model, epochs=100, lr=0.001, device='cpu'):
    """
    Trains the edge classifier model.

    @param graphs List of DGL graphs for training.
    @param model EdgeClassifier model.
    @param epochs Number of training epochs.
    @param lr Learning rate.
    @param device Device to run the training on (CPU or CUDA).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        total = 0
        
        for g in graphs:
            g = g.to(device)
            node_feats = g.ndata['feat'].to(device).float()
            edge_feats = g.edata['feat'].to(device).float()
            labels = g.edata['label'].to(device)
            train_mask = g.edata['train_mask'].to(device)
            
            optimizer.zero_grad()
            logits = model(g, node_feats, edge_feats)
            loss = loss_fn(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * train_mask.sum().item()
            
            _, predicted = torch.max(logits, dim=1)
            correct = (predicted == labels) & train_mask
            acc = correct.sum().item()
            epoch_acc += acc
            total += train_mask.sum().item()
        
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
    with torch.no_grad():
        best_val_acc = 0
        best_test_acc = 0
        
        for g in graphs:
            g = g.to(device)
            node_feats = g.ndata['feat'].to(device).float()
            edge_feats = g.edata['feat'].to(device).float()
            labels = g.edata['label'].to(device)
            
            logits = model(g, node_feats, edge_feats)
            _, predicted = torch.max(logits, dim=1)
            
            val_mask = g.edata['val_mask'].to(device)
            val_correct = (predicted == labels) & val_mask
            val_acc = val_correct.sum().item() / val_mask.sum().item()
            
            test_mask = g.edata['test_mask'].to(device)
            test_correct = (predicted == labels) & test_mask
            test_acc = test_correct.sum().item() / test_mask.sum().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
        
        print(f"Best Validation Accuracy: {best_val_acc:.4f}, Corresponding Test Accuracy: {best_test_acc:.4f}")

if __name__ == "__main__":
    nodes_df, edges_df = load_csv_data('./generated-data')
    graphs = construct_dgl_graphs(nodes_df, edges_df)
    
    sample_graph = graphs[0]
    in_feats = sample_graph.ndata['feat'].shape[1]
    edge_feat_size = sample_graph.edata['feat'].shape[1]
    num_classes = sample_graph.edata['label'].max().item() + 1
    
    hidden_size = 16
    model = EdgeClassifier(in_feats, hidden_size, num_classes, edge_feat_size=edge_feat_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_edge_classifier(graphs, model, epochs=100, lr=0.001, device=device)
    evaluate_edge_classifier(graphs, model, device=device)
