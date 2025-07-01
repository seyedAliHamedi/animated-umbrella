import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class Agent(nn.Module):
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2, lr=0.0005):
        super().__init__()

        self.conv1 = GATConv(
            num_node_features, hidden_channels1, heads=2, concat=True)
        self.conv2 = GATConv(hidden_channels1 * 2,
                             hidden_channels2, heads=1, concat=False)

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_channels2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.actor_head = nn.Sequential(
            nn.Linear(num_node_features +
                      hidden_channels2 * 2, hidden_channels2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels2, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, data):
        x_initial, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.conv1(x_initial, edge_index))
        embed = F.tanh(self.conv2(x, edge_index))

        global_embed = embed.mean(axis=0)

        state_value = self.critic_head(global_embed).squeeze(-1)

        combined_embeddings = torch.cat(
            [x_initial, embed, global_embed.repeat(
                embed.size(0), 1)], dim=1)

        actor_logits = self.actor_head(combined_embeddings).squeeze(-1)

        return actor_logits, state_value

    def dict_to_data(self, adj_matrix, node_features_dict):
        num_nodes = len(node_features_dict)

        edge_list = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i][j] > 0:
                    edge_list.append([i, j])
                    edge_list.append([j, i])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t(
        ).contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

        features = torch.zeros((num_nodes, 11), dtype=torch.float)
        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]
            features[node_id, 0] = 1.0 if node_data.get('is_client') else 0.0
            features[node_id, 1] = 1.0 if node_data.get('is_server') else 0.0
            features[node_id, 2] = node_data['P_idle']
            features[node_id, 3] = node_data['P_rx']
            features[node_id, 4] = node_data['P_tx']
            features[node_id, 5] = node_data['P_base']
            features[node_id, 6] = float(
                node_data['graph_metrics']['degree_centrality']['original'])
            features[node_id, 7] = float(
                node_data['graph_metrics']['clustering_coefficient']['original'])
            features[node_id, 8] = float(
                node_data['graph_metrics']['eigenvector_centrality']['original'])
            features[node_id, 9] = 1.0 if node_data['graph_metrics']['is_articulation_point']['original'] else 0.0
            features[node_id, 10] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['original'])

        return Data(x=features, edge_index=edge_index)

    def get_action(self, node_metrics, adj_matrix, edge_metrics):
        data = self.dict_to_data(adj_matrix, node_metrics, edge_metrics)
        logits, _ = self.forward(data)
        p = torch.sigmoid(logits)
        actions = torch.bernoulli(p)
        return actions, p, logits

    def dict_to_data2(self, adj_matrix, node_features_dict):
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
        features = torch.zeros((num_nodes, 3), dtype=torch.float)
        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]
            features[node_id, 0] = 1.0 if node_data['is_client'] else 0.0
            features[node_id, 1] = 1.0 if node_data['is_server'] else 0.0
            features[node_id, 2] = (node_id+1)/num_nodes
            continue
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
            features[node_id, 1] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['original'])
            features[node_id, 2] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['current'])

        # Normalize features (except binary features at indices 0, 9, 10)
        # for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        #     min_val = features[:, i].min()
        #     max_val = features[:, i].max()
        #     if max_val - min_val > 0:
        #         features[:, i] = (features[:, i] - min_val) / \
        #             (max_val - min_val)
        #     else:
        #         features[:, i] = 0.0  # If all values are the same

        return Data(x=features, edge_index=edge_index)
