from dgl.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import torch
import dgl.data
import dgl
import os

os.environ["DGLBACKEND"] = "pytorch"

dataset = dgl.data.CoraGraphDataset()
print(f"Number of categories: {dataset.num_classes}")

g = dataset[0]
print(f'Node Data {g.ndata}')
print(f'Edge Data {g.edata}')


class GNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'gcn')
        self.conv2 = SAGEConv(h_feats, num_classes, 'gcn')
        # self.conv3 = SAGEConv(h_feats, h_feats, 'mean')
        # self.conv4 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        # h = F.relu(h)
        # h = self.conv3(g, h)
        # h = F.relu(h)
        # h = self.conv4(g, h)
        return h


def train(g, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(1000):
        logits = model(g, features)

        pred = logits.argmax(1)
        # if e == 999:
        #     print(pred)
        #     print(logits)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
            # print(
            #     f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {
            #         test_acc:.3f} (best {best_test_acc:.3f})"
            # )
    print(f"Best Results: Val: {best_val_acc:.3f}, Test: {best_test_acc:.3f}")

# print(g.ndata["feat"].shape)
model = GNN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(g, model)
