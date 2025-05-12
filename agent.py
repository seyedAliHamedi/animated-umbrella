import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops
import numpy as np
import random


# class WeightedSelfAggregation(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')
#         self.lin = nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         self.num_nodes = x.size(0)
#         edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
#         x = self.lin(x)
#         return self.propagate(edge_index=edge_index, x=x)

#     def message(self, x_j, edge_index):
#         source, target = edge_index
#         is_self = source == target
#         N = self.num_nodes
#         weights = torch.where(is_self, torch.tensor(
#             1.0, device=x_j.device), torch.tensor(0.0 / (N - 1), device=x_j.device))
#         return x_j * weights.view(-1, 1)


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2, lr=0.001):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.embed = nn.Linear(num_node_features, hidden_channels2)
        self.nn = nn.Linear(hidden_channels2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        temp = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        temp = self.embed(temp)
        x = x + temp
        x = self.nn(x)
        return x

    def dict_to_data(self, adj_matrix, node_features_dict):
        edge_index = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] > 0:
                    edge_index.append([i, j])

        num_nodes = len(node_features_dict)
        if len(edge_index) == 0:
            edge_index = [[i, i] for i in range(num_nodes)]

        edge_index = torch.tensor(
            edge_index, dtype=torch.long).t().contiguous()

        feature_list = []
        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]
            features = [
                1.0 if node_data['is_client_server'] else 0.0,
                # ]
                # a = [
                float(node_data['graph_metrics']
                      ['betweenness_centrality']['original']),
                float(node_data['graph_metrics']
                      ['betweenness_centrality']['current']),
                float(node_data['graph_metrics']
                      ['degree_centrality']['original']),
                float(node_data['graph_metrics']
                      ['degree_centrality']['current']),
                float(node_data['graph_metrics']
                      ['clustering_coefficient']['original']),
                float(node_data['graph_metrics']
                      ['clustering_coefficient']['current']),
                float(node_data['graph_metrics']
                      ['eigenvector_centrality']['original']),
                float(node_data['graph_metrics']
                      ['eigenvector_centrality']['current']),
                1.0 if node_data['graph_metrics']['is_articulation_point']['original'] else 0.0,
                1.0 if node_data['graph_metrics']['is_articulation_point']['current'] else 0.0
            ]
            feature_list.append(features)

        x = torch.tensor(feature_list, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    def get_action(self, metrics, adj_matrix):
        data = self.dict_to_data(adj_matrix, metrics)
        logits = self(data)
        p = torch.sigmoid(logits)
        if random.random() < 0.1:
            actions = torch.bernoulli(torch.ones_like(p) * 0.5)
        else:
            actions = torch.bernoulli(p)
        return actions, p, logits
