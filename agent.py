import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler


class Agent(nn.Module):
    def __init__(self, num_node_features, node_hidden_channels, num_edge_features, edge_hidden_channels, num_nodes=4, lr=0.001):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_edges = num_nodes * (num_nodes - 1) // 2

        self.gcn1 = GCNConv(num_node_features, node_hidden_channels)
        self.node_bn = nn.BatchNorm1d(node_hidden_channels)

        self.edge_nn = nn.Sequential(
            nn.Linear(num_edge_features + 2 *
                      node_hidden_channels, edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, edge_hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels // 2, 1)
        )

        self.global_decision = nn.Sequential(
            nn.Linear(num_nodes * node_hidden_channels,
                      edge_hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels * 2, edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, self.max_edges)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def node_conv(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.node_bn(x)
        x = F.relu(x)
        return x

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_embeddings = self.node_conv(x, edge_index)

        node_embeddings_flat = node_embeddings.view(-1)

        logits = self.global_decision(node_embeddings_flat)

        return logits

    def dict_to_data(self, adj_matrix, node_features_dict, edge_features_dict):
        edge_index = []
        edge_features = []

        for i in range(len(adj_matrix)):
            for j in range(i+1, len(adj_matrix[i])):
                if adj_matrix[i][j] > 0:
                    # Add edges in both directions for undirected graph
                    edge_index.append([i, j])
                    edge_index.append([j, i])

                    edge_feature_vector = self._process_edge_features(
                        edge_features_dict.get((i, j), {}))
                    edge_features.append(edge_feature_vector)
                    edge_features.append(edge_feature_vector)

        if len(edge_index) == 0:
            for i in range(len(node_features_dict)):
                edge_index.append([i, i])
                edge_features.append(torch.zeros(11, dtype=torch.float))

        edge_index = torch.tensor(
            edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features) if edge_features else torch.zeros(
            (0, 11), dtype=torch.float)

        feature_list = []
        for node_id in range(len(node_features_dict)):
            node_data = node_features_dict[node_id]
            features = [
                1.0 if node_data['is_active'] else 0.0,
                1.0 if node_data['is_client_server'] else 0.0,
                float(node_data['avg_power_per_operation']),
                float(node_data['avg_energy_consumption']),
                float(node_data['idle_interface_energy']),
                float(node_data['tx_packets']),
                float(node_data['tx_bytes']),
                float(node_data['rx_packets']),
                float(node_data['rx_bytes']),
                float(node_data['lost_on_send']),
                float(node_data['lost_on_receive']),
                float(node_data['total_transmit_time']),
                float(node_data['avg_transmit_time']),
                float(node_data['total_receive_time']),
                float(node_data['avg_receive_time']),
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

        features_array = np.array(feature_list, dtype=np.float32)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        x = torch.tensor(normalized_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _process_edge_features(self, edge_data):
        if not edge_data:
            return torch.zeros(11, dtype=torch.float)

        link_type_numeric = 1.0 if edge_data.get(
            'link_type', '') == 'p2p' else 0.0

        rate_value = float(edge_data.get('link_rate', '0').replace('Mbps', ''))
        delay_value = float(edge_data.get('link_delay', '0').replace('ms', ''))

        features = [
            float(edge_data.get('is_active', 0)),
            float(edge_data.get('traffic_volume', 0)),
            float(edge_data.get('bytes_transferred', 0)),
            float(edge_data.get('avg_latency', 0)),
            float(edge_data.get('max_latency', 0)),
            float(edge_data.get('packet_loss_rate', 0)),
            link_type_numeric,
            rate_value,
            delay_value,
            float(edge_data.get('link_queue_size', 0)),
            float(edge_data.get('link_error_rate', 0)),
        ]

        return torch.tensor(features, dtype=torch.float)

    def get_action(self, node_metrics, adj_matrix, edge_metrics):
        data = self.dict_to_data(adj_matrix, node_metrics, edge_metrics)

        logits = self(data)

        p = torch.sigmoid(logits)

        actions = torch.bernoulli(p)

        edge_index = self._create_full_edge_index(len(adj_matrix))

        return actions, p, edge_index

    def _create_full_edge_index(self, num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edge_index.append([i, j])

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
