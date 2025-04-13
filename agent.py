import numpy as np

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch_geometric.nn import GCNConv


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.lin1 = nn.Linear(hidden_channels2, 64)
        self.lin2 = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def dict_to_data(self, adj_matrix, node_features_dict):
        edge_index = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] > 0:
                    edge_index.append([i, j])

        # Check if no edge is found in the adjacency matrix.
        num_nodes = len(node_features_dict)
        if len(edge_index) == 0:
            # If no edges are present, add self-loops for all nodes.
            edge_index = [[i, i] for i in range(num_nodes)]
            print("No edges found. Added self-loops for all nodes.")

        edge_index = torch.tensor(
            edge_index, dtype=torch.long).t().contiguous()

        feature_list = []
        for node_id in range(num_nodes):
            node_data = node_features_dict[str(node_id)]
            features = [
                1.0 if node_data['is_active'] else 0.0,
                node_data['avg_power_per_operation'],
                node_data['avg_energy_consumption'],
                node_data['idle_interface_energy'],
                node_data['tx_packets'],
                node_data['tx_bytes'],
                node_data['rx_packets'],
                node_data['rx_bytes'],
                node_data['lost_on_send'],
                node_data['lost_on_receive'],
                node_data['expected_packets']['high_priority'],
                node_data['expected_packets']['medium_priority'],
                node_data['expected_packets']['low_priority'],
                node_data['active_interfaces'],
                node_data['graph_metrics']['betweenness_centrality']['original'],
                node_data['graph_metrics']['betweenness_centrality']['current'],
                node_data['graph_metrics']['degree_centrality']['original'],
                node_data['graph_metrics']['degree_centrality']['current'],
                node_data['graph_metrics']['clustering_coefficient']['original'],
                node_data['graph_metrics']['clustering_coefficient']['current'],
                node_data['graph_metrics']['eigenvector_centrality']['original'],
                node_data['graph_metrics']['eigenvector_centrality']['current'],
                1.0 if node_data['graph_metrics']['is_articulation_point']['original'] else 0.0,
                1.0 if node_data['graph_metrics']['is_articulation_point']['current'] else 0.0
            ]
            feature_list.append(features)

        features_array = np.array(feature_list)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        x = torch.tensor(normalized_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        return data

    def get_action(self, metrics, adj_matrix):
        data = self.dict_to_data(adj_matrix, metrics)
        logits = self(data)  # Shape: (num_nodes, 1)
        # Compute probabilities using sigmoid.
        p = torch.sigmoid(logits).view(-1)
        # Sample stochastically from the Bernoulli distribution.
        actions = torch.bernoulli(p)
        print("Sigmoid probabilities:", p)
        print("Sampled actions:", actions)
        return actions, p, logits
