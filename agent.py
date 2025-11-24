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
    def __init__(self, num_node_features, hidden_channels1, hidden_channels2, lr=0.0005):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels1)
        # self.conv2 = GATConv(hidden_channels1, hidden_channels2)
        # self.conv3 = GATConv(hidden_channels1, hidden_channels2)
        # self.conv4 = GATConv(hidden_channels1, hidden_channels2)
        # self.conv5 = GATConv(hidden_channels1, hidden_channels2)
        self.embed1 = nn.Linear(num_node_features, hidden_channels1)
        self.embed2 = nn.Linear(hidden_channels1, hidden_channels1)
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
        temp = F.relu(self.embed1(temp))
        temp = F.relu(self.embed2(temp))
        x = torch.cat([x, temp], dim=1)
        x = self.nn(x)
        # x = self.nn2(x)
        # x = self.nn3(x)
        return x

    def dict_to_data(self, adj_matrix, node_features_dict):
        router_type = {
            i: sample_data["mawi_routers_one_per_city"][i]
            for i in range(len(adj_matrix))
        }
        # router_type = {0: sample_data["routers"][4],
        #                1: sample_data["routers"][9],
        #                2: sample_data["routers"][1],
        #                3: sample_data["routers"][8],
        #                4: sample_data["routers"][12],
        #                5: sample_data["routers"][12]}

        # router_type = {0: sample_data["routers"][4],
        #                1: sample_data["routers"][7],
        #                2: sample_data["routers"][13],
        #                3: sample_data["routers"][8],
        #                4: sample_data["routers"][4],
        #                5: sample_data["routers"][6],
        #                6: sample_data["routers"][1],
        #                7: sample_data["routers"][3], }

        all_routers = list(router_type.values())
        max_p_idle, max_p_rx, max_p_tx, max_p_base, max_queue, max_err = (
            max(router[key] for router in all_routers)
            for key in ["P_idle", "P_rx", "P_tx", "P_base", "Queue_size_packets", "Avg_loss_percent"]
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

        features = torch.zeros((num_nodes, 18), dtype=torch.float)
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
                features[node_id, 7] = float(
                    router_type[node_id]["Queue_size_packets"]) / max_queue
                features[node_id, 8] = float(
                    router_type[node_id]["Avg_loss_percent"]) / max_err

            else:
                features[node_id, 3] = 0.0
                features[node_id, 4] = 0.0
                features[node_id, 5] = 0.0
                features[node_id, 6] = 0.0
                features[node_id, 7] = 0.0
                features[node_id, 8] = 0.0

            # RTT features (already normalized from collect_graph_metrics)
            features[node_id, 9] = float(
                node_data['graph_metrics'].get('avg_rtt_neigh', 0.5))
            features[node_id, 10] = float(
                node_data['graph_metrics'].get('min_rtt_neigh', 0.5))
            features[node_id, 11] = float(
                node_data['graph_metrics'].get('max_rtt_neigh', 0.5))
            features[node_id, 12] = float(
                node_data['graph_metrics'].get('min_rtt_to_src', 1.0))
            features[node_id, 13] = float(
                node_data['graph_metrics'].get('max_rtt_to_src', 1.0))
            features[node_id, 14] = float(
                node_data['graph_metrics'].get('rtt_ratio_src', 1.0))
            features[node_id, 15] = float(
                node_data['graph_metrics'].get('min_rtt_to_dst', 1.0))
            features[node_id, 16] = float(
                node_data['graph_metrics'].get('max_rtt_to_dst', 1.0))
            features[node_id, 17] = float(
                node_data['graph_metrics'].get('rtt_ratio_dst', 1.0))

        return Data(x=features, edge_index=edge_index)

    def get_action(self, metrics, adj_matrix):
        data = self.dict_to_data(adj_matrix, metrics)
        logits = self(data)
        p = torch.sigmoid(logits)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        # if random.random() < 0.1:
        #     actions = torch.bernoulli(torch.ones_like(p) * 0.5)
        # else:
        actions = torch.bernoulli(p)
        return actions, p, logits


class TrafficAwareAgent(nn.Module):
    """
    Traffic-Aware GNN Agent with Cross-Attention.

    Dual pipeline architecture:
    - Graph Pipeline: GAT for node embeddings
    - Traffic Pipeline: MLP encoder for QoS type features
    - Cross-Attention: Nodes query traffic embeddings to learn which QoS types matter
    """

    def __init__(
        self,
        num_node_features=18,
        hidden=64,
        n_qos_types=4,
        traffic_feat_per_qos=8,
        n_heads=4,
        lr=0.0005
    ):
        super().__init__()

        self.n_qos_types = n_qos_types
        self.hidden = hidden

        # === Graph Pipeline ===
        self.gat = GATConv(num_node_features, hidden)

        self.residual = nn.Sequential(
            nn.Linear(num_node_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # === Traffic Pipeline ===
        # Shared encoder applied to each QoS type
        self.traffic_encoder = nn.Sequential(
            nn.Linear(traffic_feat_per_qos, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # === Cross-Attention ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            batch_first=True
        )

        # === Decision Head ===
        # Input: node_emb (GAT) + residual (MLP) + attended (cross-attention) = hidden * 3
        self.decision = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def encode_traffic(self, traffic_data):
        """
        Encode each QoS type separately.

        Args:
            traffic_data: [4, 8] tensor (4 QoS types, 8 features each)

        Returns:
            [4, hidden] tensor of traffic embeddings
        """
        return self.traffic_encoder(traffic_data)

    def forward(self, data, traffic_data):
        """
        Forward pass with graph and traffic inputs.

        Args:
            data: PyG Data object with x [N, 18] and edge_index
            traffic_data: [4, 8] tensor of traffic features

        Returns:
            logits: [N, 1] action logits
            attn_weights: [N, 4] attention over QoS types per node
        """
        x, edge_index = data.x, data.edge_index

        # === Graph Pipeline ===
        node_emb = F.relu(self.gat(x, edge_index))  # [N, hidden]
        residual = self.residual(x)                  # [N, hidden]

        # === Traffic Pipeline ===
        traffic_emb = self.encode_traffic(traffic_data)  # [4, hidden]

        # === Cross-Attention ===
        # Add batch dimension for attention
        node_emb_batch = node_emb.unsqueeze(0)       # [1, N, hidden]
        traffic_emb_batch = traffic_emb.unsqueeze(0) # [1, 4, hidden]

        attended, attn_weights = self.cross_attn(
            query=node_emb_batch,    # Nodes ask: "What traffic should I consider?"
            key=traffic_emb_batch,   # Traffic answers based on similarity
            value=traffic_emb_batch  # Traffic provides its information
        )
        # attended: [1, N, hidden]
        # attn_weights: [1, N, 4]

        attended = attended.squeeze(0)           # [N, hidden]
        attn_weights = attn_weights.squeeze(0)   # [N, 4]

        # === Combine & Decide ===
        # Include all three: GAT output (graph structure) + residual (raw features) + attended (traffic-aware)
        combined = torch.cat([node_emb, residual, attended], dim=1)  # [N, hidden*3]
        logits = self.decision(combined)                              # [N, 1]

        return logits, attn_weights

    def dict_to_data(self, adj_matrix, node_features_dict):
        """Convert node features dict to PyG Data object with 22 features (including per-QoS FBC)."""
        router_type = {
            i: sample_data["mawi_routers_one_per_city"][i]
            for i in range(len(adj_matrix))
        }

        all_routers = list(router_type.values())
        max_p_idle, max_p_rx, max_p_tx, max_p_base, max_queue, max_err = (
            max(router[key] for router in all_routers)
            for key in ["P_idle", "P_rx", "P_tx", "P_base", "Queue_size_packets", "Avg_loss_percent"]
        )

        edge_list = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] > 0:
                    edge_list.append([i, j])

        num_nodes = len(node_features_dict)
        if not edge_list:
            edge_list = [[i, i] for i in range(num_nodes)]

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # 22 features: 18 original + 4 per-QoS FBC
        features = torch.zeros((num_nodes, 22), dtype=torch.float)
        for node_id in range(num_nodes):
            node_data = node_features_dict[node_id]

            features[node_id, 0] = 1.0 if node_data['is_client_server'] else 0.0
            features[node_id, 1] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['original'])
            features[node_id, 2] = float(
                node_data['graph_metrics']['flow_betweenness_centrality']['current'])

            if node_id in router_type:
                features[node_id, 3] = float(router_type[node_id]["P_idle"]) / max_p_idle
                features[node_id, 4] = float(router_type[node_id]["P_rx"]) / max_p_rx
                features[node_id, 5] = float(router_type[node_id]["P_tx"]) / max_p_tx
                features[node_id, 6] = float(router_type[node_id]["P_base"]) / max_p_base
                features[node_id, 7] = float(router_type[node_id]["Queue_size_packets"]) / max_queue
                features[node_id, 8] = float(router_type[node_id]["Avg_loss_percent"]) / max_err
            else:
                features[node_id, 3:9] = 0.0

            features[node_id, 9] = float(node_data['graph_metrics'].get('avg_rtt_neigh', 0.5))
            features[node_id, 10] = float(node_data['graph_metrics'].get('min_rtt_neigh', 0.5))
            features[node_id, 11] = float(node_data['graph_metrics'].get('max_rtt_neigh', 0.5))
            features[node_id, 12] = float(node_data['graph_metrics'].get('min_rtt_to_src', 1.0))
            features[node_id, 13] = float(node_data['graph_metrics'].get('max_rtt_to_src', 1.0))
            features[node_id, 14] = float(node_data['graph_metrics'].get('rtt_ratio_src', 1.0))
            features[node_id, 15] = float(node_data['graph_metrics'].get('min_rtt_to_dst', 1.0))
            features[node_id, 16] = float(node_data['graph_metrics'].get('max_rtt_to_dst', 1.0))
            features[node_id, 17] = float(node_data['graph_metrics'].get('rtt_ratio_dst', 1.0))

            # Per-QoS FBC features (4 new features at indices 18-21)
            features[node_id, 18] = float(node_data['graph_metrics'].get('fbc_Interactive_Web', 0.0))
            features[node_id, 19] = float(node_data['graph_metrics'].get('fbc_Streaming_Media', 0.0))
            features[node_id, 20] = float(node_data['graph_metrics'].get('fbc_Background_Sync', 0.0))
            features[node_id, 21] = float(node_data['graph_metrics'].get('fbc_Real_Time_Interactive', 0.0))

        return Data(x=features, edge_index=edge_index)

    def get_action(self, metrics, adj_matrix, traffic_data):
        """
        Sample actions from policy with traffic awareness.

        Args:
            metrics: Node features dict
            adj_matrix: Current adjacency matrix
            traffic_data: [4, 8] tensor of normalized traffic features

        Returns:
            actions: [N] binary tensor
            probs: [N, 1] action probabilities
            logits: [N, 1] raw logits
            attn_weights: [N, 4] attention weights for interpretability
        """
        data = self.dict_to_data(adj_matrix, metrics)
        logits, attn_weights = self.forward(data, traffic_data)
        p = torch.sigmoid(logits)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        actions = torch.bernoulli(p)

        return actions, p, logits, attn_weights
