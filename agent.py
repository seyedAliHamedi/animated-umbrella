import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures




class Agent(nn.Module):
    def __init__(self, num_node_features,hidden_channels1,hidden_channels2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels1)
        self.conv2 = GCNConv( hidden_channels1,hidden_channels2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        print(x.shape)

        return x

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
model = Agent(num_node_features=dataset.num_node_features,hidden_channels1=16,hidden_channels2=32)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
data = dataset[0]  
print(data)


for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d} | Loss: {loss:.4f}')

# Evaluation
model.eval()
out = model(data)
pred = out.argmax(dim=1)
correct = pred[data.test_mask] == data.y[data.test_mask]
acc = int(correct.sum()) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
