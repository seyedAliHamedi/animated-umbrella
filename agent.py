import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops
import numpy as np
import random


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2, lr=0.001):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels1)
        self.conv2 = GATConv(hidden_channels1, hidden_channels2)
        self.embed = nn.Linear(num_node_features, hidden_channels2)
        self.nn = nn.Linear(hidden_channels2 * 2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        temp = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        temp = self.embed(temp)
        x = torch.cat([x, temp], dim=1)
        x = self.nn(x)
        return x

    def dict_to_data(self, adj_matrix, node_features_dict):
        # Pre-create edge list
        edge_list = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] > 0:
                    edge_list.append([i, j])

        num_nodes = len(node_features_dict)
        if not edge_list:
            edge_list = [[i, i] for i in range(num_nodes)]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Batch process features
        features = torch.zeros((num_nodes, 11), dtype=torch.float)
        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]
            features[node_id, 0] = 1.0 if node_data['is_client_server'] else 0.0
            features[node_id, 1] = float(
                node_data['graph_metrics']['betweenness_centrality']['original'])
            features[node_id, 2] = float(
                node_data['graph_metrics']['betweenness_centrality']['current'])
            features[node_id, 3] = float(
                node_data['graph_metrics']['degree_centrality']['original'])
            features[node_id, 4] = float(
                node_data['graph_metrics']['degree_centrality']['current'])
            features[node_id, 5] = float(
                node_data['graph_metrics']['clustering_coefficient']['original'])
            features[node_id, 6] = float(
                node_data['graph_metrics']['clustering_coefficient']['current'])
            features[node_id, 7] = float(
                node_data['graph_metrics']['eigenvector_centrality']['original'])
            features[node_id, 8] = float(
                node_data['graph_metrics']['eigenvector_centrality']['current'])
            features[node_id, 9] = 1.0 if node_data['graph_metrics']['is_articulation_point']['original'] else 0.0
            features[node_id, 10] = 1.0 if node_data['graph_metrics']['is_articulation_point']['current'] else 0.0

        # Normalize features (except binary features at indices 0, 9, 10)
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            min_val = features[:, i].min()
            max_val = features[:, i].max()
            if max_val - min_val > 0:
                features[:, i] = (features[:, i] - min_val) / \
                    (max_val - min_val)
            else:
                features[:, i] = 0.0  # If all values are the same

        return Data(x=features, edge_index=edge_index)

    def get_action(self, metrics, adj_matrix):
        data = self.dict_to_data(adj_matrix, metrics)
        logits = self(data)
        p = torch.sigmoid(logits)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        if random.random() < 0.1:
            actions = torch.bernoulli(torch.ones_like(p) * 0.5)
        else:
            actions = torch.bernoulli(p)
        return actions, p, logits
