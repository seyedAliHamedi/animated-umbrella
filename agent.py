import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2, lr=0.001):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.lin1 = nn.Linear(hidden_channels2, 64)
        self.lin2 = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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

        num_nodes = len(node_features_dict)
        if len(edge_index) == 0:
            edge_index = [[i, i] for i in range(num_nodes)]
            print("Warning: No edges found. Added self-loops for all nodes.")

        edge_index = torch.tensor(
            edge_index, dtype=torch.long).t().contiguous()

        feature_list = []

        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]
            features = [
                1.0 if node_data['is_active'] else 0.0,
                float(node_data['avg_power_per_operation']),
                float(node_data['avg_energy_consumption']),
                float(node_data['idle_interface_energy']),
                float(node_data['tx_packets']),
                float(node_data['tx_bytes']),
                float(node_data['rx_packets']),
                float(node_data['rx_bytes']),
                float(node_data['lost_on_send']),
                float(node_data['lost_on_receive']),
                float(node_data['expected_packets']['high_priority']),
                float(node_data['expected_packets']['medium_priority']),
                float(node_data['expected_packets']['low_priority']),
                float(node_data['active_interfaces']),
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

        features_array = np.array(feature_list)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        x = torch.tensor(normalized_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def get_action(self, metrics, adj_matrix):
        data = self.dict_to_data(adj_matrix, metrics)

        logits = self(data)
        p = torch.sigmoid(logits).view(-1)
        actions = torch.bernoulli(p)

        return actions, p, logits
