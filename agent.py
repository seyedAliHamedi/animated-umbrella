import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops
import numpy as np
import random
from sim.utils import sample_data


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2, lr=0.001):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels1)
        # self.conv2 = GATConv(hidden_channels1, hidden_channels2)
        # self.conv3 = GATConv(hidden_channels1, hidden_channels2)
        # self.conv4 = GATConv(hidden_channels1, hidden_channels2)
        # self.conv5 = GATConv(hidden_channels1, hidden_channels2)
        self.embed = nn.Linear(num_node_features, hidden_channels1)
        # self.nn1 = nn.Linear(hidden_channels1 * 2, 64)
        # self.nn2 = nn.Linear(128, 64)
        # self.nn2 = nn.Linear(64, 1)
        self.nn = nn.Linear(hidden_channels1 * 2, 1)
        # self.nn = nn.Linear(hidden_channels2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        temp = x
        x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        temp = self.embed(temp)
        x = torch.cat([x, temp], dim=1)
        x = self.nn(x)
        # x = self.nn2(x)
        # x = self.nn3(x)
        return x

    def dict_to_data(self, adj_matrix, node_features_dict):
        # router_type = {
        #     i: sample_data["routers"][i % len(sample_data["routers"])]
        #     for i in range(len(adj_matrix))
        # }
        # router_type = {0: sample_data["routers"][4],
        #                1: sample_data["routers"][9],
        #                2: sample_data["routers"][1],
        #                3: sample_data["routers"][8],
        #                4: sample_data["routers"][12],
        #                5: sample_data["routers"][12]}
        router_type = {0: sample_data["routers"][4],
                       1: sample_data["routers"][7],
                       2: sample_data["routers"][13],
                       3: sample_data["routers"][8],
                       4: sample_data["routers"][4],
                       5: sample_data["routers"][6],
                       6: sample_data["routers"][1],
                       7: sample_data["routers"][3], }

        all_routers = list(router_type.values())
        max_p_idle, max_p_rx, max_p_tx, max_p_base = (
            max(router[key] for router in all_routers)
            for key in ["P_idle", "P_rx", "P_tx", "P_base"]
        )
        # Pre-create edge list
        edge_list = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] > 0:
                    edge_list.append([i, j])
                    # edge_list.append([j, i])

        num_nodes = len(node_features_dict)
        if not edge_list:
            edge_list = [[i, i] for i in range(num_nodes)]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        features = torch.zeros((num_nodes, 16), dtype=torch.float)
        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]

            # Original features
            features[node_id, 0] = 1.0 if node_data['is_client_server'] else 0.0
            features[node_id, 1] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['original'])
            features[node_id, 2] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['current'])

            # Router power features
            if node_id in router_type:
                features[node_id, 3] = float(
                    router_type[node_id]["P_idle"]) / max_p_idle
                features[node_id, 4] = float(
                    router_type[node_id]["P_rx"]) / max_p_rx
                features[node_id, 5] = float(
                    router_type[node_id]["P_tx"]) / max_p_tx
                features[node_id, 6] = float(
                    router_type[node_id]["P_base"]) / max_p_base
            else:
                features[node_id, 3] = 0.0
                features[node_id, 4] = 0.0
                features[node_id, 5] = 0.0
                features[node_id, 6] = 0.0

            # RTT features (already normalized from collect_graph_metrics)
            features[node_id, 7] = float(
                node_data['graph_metrics'].get('avg_rtt_neigh', 0.5))
            features[node_id, 8] = float(
                node_data['graph_metrics'].get('min_rtt_neigh', 0.5))
            features[node_id, 9] = float(
                node_data['graph_metrics'].get('max_rtt_neigh', 0.5))
            features[node_id, 10] = float(
                node_data['graph_metrics'].get('min_rtt_to_src', 1.0))
            features[node_id, 11] = float(
                node_data['graph_metrics'].get('max_rtt_to_src', 1.0))
            features[node_id, 12] = float(
                node_data['graph_metrics'].get('rtt_ratio_src', 1.0))
            features[node_id, 13] = float(
                node_data['graph_metrics'].get('min_rtt_to_dst', 1.0))
            features[node_id, 14] = float(
                node_data['graph_metrics'].get('max_rtt_to_dst', 1.0))
            features[node_id, 15] = float(
                node_data['graph_metrics'].get('rtt_ratio_dst', 1.0))

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
