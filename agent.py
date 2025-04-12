import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.lin1 = nn.Linear(hidden_channels2, 64)
        self.lin2 = nn.Linear(64, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        x = F.sigmoid(x)
        return x
