# Network Optimization Architecture

## Table of Contents
1. [Overview](#overview)
2. [Base Implementation](#base-implementation)
3. [Traffic-Aware Extensions](#traffic-aware-extensions)
4. [Training Infrastructure](#training-infrastructure)

---

# Overview

This project implements a **Traffic-Aware Reinforcement Learning agent** that optimizes network topology by selectively deactivating routers to minimize energy consumption while maintaining Quality of Service (QoS). The system uses **Graph Attention Networks (GAT)** with **cross-attention over traffic features** to make topology decisions informed by both graph structure and traffic characteristics.

**Key Features:**
- Graph Neural Network (GAT) for topology understanding
- Cross-attention mechanism for traffic-aware decision making
- Multi-objective optimization (energy, QoS, efficiency)
- NS-3 packet-level simulation for realistic validation
- Real-world MAWI traffic data integration

---

# Base Implementation

## Objective

Learn to **deactivate network routers** to minimize energy consumption while maintaining connectivity and Quality of Service (QoS).

This is a **multi-objective optimization** problem balancing:
- **Energy consumption** (minimize)
- **Quality of Service** (maximize)
- **Router efficiency** (use only necessary routers)

---

## Network Environment

### Topology
- **13 nodes** representing real MAWI (Measurement and Analysis on the WIDE Internet) infrastructure
- Japanese network locations: NTT Otemachi, KDDI Otemachi, Dojima, etc.
- Adjacency matrix defines physical connectivity

### Simulation
- **NS-3 network simulator** with Python bindings
- Packet-level simulation (not flow-level approximation)
- **RIP routing protocol** for dynamic path computation
- Simulates real packet transmission, queuing, delays, losses

### Data Flow
```
Clients → Gateway Routers → [Network Topology] → Gateway Routers → Servers
```

---

## RL Agent Architecture

### Model: Graph Attention Network (GAT) + MLP

```
Input: Node Features [N × 18]
           ↓
    ┌──────┴──────┐
    ↓             ↓
  GATConv      Linear
    ↓             ↓
   ReLU        ReLU
    ↓             ↓
    │          Linear
    │             ↓
    │           ReLU
    └──────┬──────┘
           ↓
       Concat [N × 128]
           ↓
        Linear
           ↓
       Logits [N × 1]
           ↓
        Sigmoid
           ↓
       Bernoulli Sample
           ↓
    Actions [N] (0 = deactivate, 1 = keep)
```

### Node Features (18 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `is_client_server` | Binary: is this a gateway node? |
| 1 | `fbc_original` | Flow Betweenness Centrality (original topology) |
| 2 | `fbc_current` | Flow Betweenness Centrality (current topology) |
| 3 | `P_idle` | Router idle power consumption (normalized) |
| 4 | `P_rx` | Router receive power (normalized) |
| 5 | `P_tx` | Router transmit power (normalized) |
| 6 | `P_base` | Router base power (normalized) |
| 7 | `Queue_size` | Router queue capacity (normalized) |
| 8 | `Avg_loss_percent` | Router average packet loss (normalized) |
| 9 | `avg_rtt_neigh` | Average RTT to neighbors |
| 10 | `min_rtt_neigh` | Minimum RTT to neighbors |
| 11 | `max_rtt_neigh` | Maximum RTT to neighbors |
| 12 | `min_rtt_to_src` | Minimum RTT to source gateway |
| 13 | `max_rtt_to_src` | Maximum RTT to source gateway |
| 14 | `rtt_ratio_src` | RTT ratio to source |
| 15 | `min_rtt_to_dst` | Minimum RTT to destination gateway |
| 16 | `max_rtt_to_dst` | Maximum RTT to destination gateway |
| 17 | `rtt_ratio_dst` | RTT ratio to destination |

---

## Reward Function

### Formula
```python
if failed_packets > 0:
    reward = -1 × (n_failed / n_total)      # Penalize broken paths
elif all_routers_active and r ≠ 1:
    reward = -1                              # Penalize doing nothing useful
else:
    reward = exp(-e_norm) × exp(q) × exp(1-r)
```

### Components

| Symbol | Meaning | Goal |
|--------|---------|------|
| `e_norm` | Normalized energy consumption | Lower is better |
| `q` | Weighted QoS score | Higher is better |
| `r` | `active_routers / path_routers` | Closer to 1 is better |

### Energy Calculation
```python
total_energy = Σ (P_base × duration) + Σ (P_tx × t_tx + P_rx × t_rx + P_idle × t_idle)
                 ↑ per active router      ↑ per active interface
```

### QoS Calculation
```python
per_flow_qos = 1 - (w_delay × d + w_jitter × j + w_loss × l)

where:
    d = min(1.0, mean_delay / sla_delay)
    j = min(1.0, mean_jitter / sla_jitter)
    l = min(1.0, lost_packets / (n_tx × sla_loss))

weighted_qos = Σ(w_i × q_i) / Σ(w_i)
    where w_i = n_rx × priority
```

---

## Training

### Algorithm
**REINFORCE** (Policy Gradient) with entropy regularization

```python
log_prob = log(p) × action + log(1-p) × (1-action)
entropy = -(p × log(p) + (1-p) × log(1-p))
loss = -Σ(log_prob × reward) - entropy_weight × Σ(entropy)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0005 |
| Entropy weight | 0.01 |
| Gradient clip | 0.5 |
| Optimizer | Adam |
| Episodes per checkpoint | 100 |

---

## Simulation Pipeline

```
1. get_state()
   └── Collect node features (centrality, power specs, RTT)

2. agent.get_action(state)
   └── GAT forward pass → Bernoulli sample → binary actions

3. changeAdj(actions, original_adj)
   └── Zero out rows/cols for deactivated nodes

4. NetworkEnv.step()
   ├── Build NS-3 topology from adjacency matrix
   ├── Install RIP routing
   ├── Create client/server applications
   ├── Run simulation (50s default)
   ├── Collect packet logs
   ├── Calculate energy, QoS, reward
   └── Return metrics

5. Policy gradient update
   └── loss.backward() → optimizer.step()
```

---

# Traffic-Aware Extensions

## Motivation & Implementation

The base model optimizes topology based only on **structure** (graph metrics, router specs). The traffic-aware extension addresses these limitations by:

1. **Prioritizing QoS types**: Real-time traffic receives different treatment than bulk transfer
2. **Adapting to traffic load**: High traffic → distribute across paths; Low traffic → minimize active routers
3. **Understanding traffic requirements**: Different SLAs drive different routing strategies

**Status**: ✅ **IMPLEMENTED** - The TrafficAwareAgent is the default agent used in training (`USE_TRAFFIC_AWARE = True` in Main.py)

---

## Architecture: Dual Pipeline with Cross-Attention

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAFFIC-AWARE AGENT                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   GRAPH PIPELINE                    TRAFFIC PIPELINE                │
│   ══════════════                    ════════════════                │
│                                                                     │
│   Node Features [N×18]              Traffic Features [4×8]          │
│         │                           (one per QoS type)              │
│         ↓                                  │                        │
│      GATConv                               ↓                        │
│         │                           ┌──────────────┐                │
│         ↓                           │   Shared     │                │
│       ReLU                          │   Encoder    │                │
│         │                           │   (MLP)      │                │
│         ↓                           └──────────────┘                │
│   Node Embeddings                          │                        │
│      [N × 64]                              ↓                        │
│         │                           Traffic Embeddings              │
│         │                              [4 × 64]                     │
│         │                                  │                        │
│         └──────────┬───────────────────────┘                        │
│                    ↓                                                │
│            ╔═══════════════════╗                                    │
│            ║  CROSS-ATTENTION  ║                                    │
│            ║                   ║                                    │
│            ║  Q = Nodes        ║                                    │
│            ║  K, V = Traffic   ║                                    │
│            ╚═══════════════════╝                                    │
│                    │                                                │
│                    ↓                                                │
│            Attended Features [N × 64]                               │
│            + Attention Weights [N × 4]                              │
│                    │                                                │
│         ┌─────────┴─────────┐                                       │
│         ↓                   ↓                                       │
│   Residual Path      Attended Path                                  │
│    (from input)                                                     │
│         └─────────┬─────────┘                                       │
│                   ↓                                                 │
│               Concat [N × 128]                                      │
│                   ↓                                                 │
│             Decision MLP                                            │
│                   ↓                                                 │
│             Logits [N × 1]                                          │
│                   ↓                                                 │
│              Sigmoid → Bernoulli                                    │
│                   ↓                                                 │
│             Actions [N]                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Traffic Features (Per QoS Type)

For each of the 4 QoS types, we extract 8 features:

| Index | Feature | Source | Meaning |
|-------|---------|--------|---------|
| 0 | `n_flows` | runtime | Number of flows of this type |
| 1 | `n_packets` | runtime | Total expected packets |
| 2 | `avg_packet_size` | config | Average bytes per packet |
| 3 | `packet_rate` | `1/interval` | Packets per second |
| 4 | `w_delay` | config | Weight for delay in QoS formula |
| 5 | `w_jitter` | config | Weight for jitter in QoS formula |
| 6 | `w_loss` | config | Weight for loss in QoS formula |
| 7 | `strictness` | derived | `1/sla_delay` (inverse = stricter) |

**Total: 4 QoS types × 8 features = 32 traffic features**

### Feature Normalization

```python
traffic_features[q_type] = [
    n_flows / max_expected_flows,           # [0, 1]
    n_packets / max_expected_packets,       # [0, 1]
    avg_packet_size / max_packet_size,      # [0, 1]
    (1/interval) / max_packet_rate,         # [0, 1]
    w_delay,                                # already [0, 1]
    w_jitter,                               # already [0, 1]
    w_loss,                                 # already [0, 1]
    (1/sla_delay) / max_strictness,         # [0, 1]
]
```

---

## Cross-Attention Mechanism

### Purpose
Learn which **QoS types** are most relevant for each **node's** decision.

### Mechanism
```
Query (Q):  Node embeddings     [N × 64]  - "What does each node need?"
Key (K):    Traffic embeddings  [4 × 64]  - "What does each QoS type offer?"
Value (V):  Traffic embeddings  [4 × 64]  - "What information to retrieve?"

Attention weights: [N × 4]
- attention[node_i, qos_j] = how much node i should consider QoS type j

Output: [N × 64]
- Weighted combination of traffic embeddings per node
```

### Interpretability
The attention weights reveal **which QoS types influence each node**:
```
Node 3: [0.6, 0.2, 0.1, 0.1]  → "Node 3 is critical for QoS type 0 (real-time)"
Node 7: [0.1, 0.1, 0.7, 0.1]  → "Node 7 matters for QoS type 2 (bulk transfer)"
```

---

## PyTorch Implementation

```python
class TrafficAwareAgent(nn.Module):
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

        # ═══ Graph Pipeline ═══
        self.gat = GATConv(num_node_features, hidden)

        self.residual = nn.Sequential(
            nn.Linear(num_node_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # ═══ Traffic Pipeline ═══
        # Shared encoder applied to each QoS type
        self.traffic_encoder = nn.Sequential(
            nn.Linear(traffic_feat_per_qos, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # ═══ Cross-Attention ═══
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            batch_first=True
        )

        # ═══ Decision Head ═══
        self.decision = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
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
        return self.traffic_encoder(traffic_data)  # [4, hidden]

    def forward(self, graph_data, traffic_data):
        """
        Forward pass with graph and traffic inputs.

        Args:
            graph_data: PyG Data object with x [N, 18] and edge_index
            traffic_data: [4, 8] tensor of traffic features

        Returns:
            logits: [N, 1] action logits
            attn_weights: [N, 4] attention over QoS types per node
        """
        x, edge_index = graph_data.x, graph_data.edge_index
        N = x.size(0)

        # ═══ Graph Pipeline ═══
        node_emb = F.relu(self.gat(x, edge_index))  # [N, hidden]
        residual = self.residual(x)                  # [N, hidden]

        # ═══ Traffic Pipeline ═══
        traffic_emb = self.encode_traffic(traffic_data)  # [4, hidden]

        # ═══ Cross-Attention ═══
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

        # ═══ Combine & Decide ═══
        combined = torch.cat([residual, attended], dim=1)  # [N, hidden*2]
        logits = self.decision(combined)                    # [N, 1]

        return logits, attn_weights

    def get_action(self, graph_data, traffic_data):
        """
        Sample actions from policy.

        Returns:
            actions: [N] binary tensor
            probs: [N, 1] action probabilities
            logits: [N, 1] raw logits
            attn_weights: [N, 4] for interpretability
        """
        logits, attn_weights = self.forward(graph_data, traffic_data)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        actions = torch.bernoulli(probs)

        return actions, probs, logits, attn_weights
```

---

## Data Flow & Integration

### Traffic Feature Extraction

Traffic features are extracted from the MAWI CSV at each episode:

```python
# In Main.py training loop
flows_by_qos = extract_flows_by_qos(row, client_gateways, server_gateways)
traffic_features = get_traffic_features(row, traffic_norm_constants, qos_profiles)
actions, p, logits, attn_weights = agent.get_action(m, adj_matrix, traffic_features)
```

### Normalization Constants

Traffic features are normalized using pre-computed 99th percentile values:
- **File**: `traffic_norm_constants.json`
- **Auto-generation**: If missing, automatically computed from CSV via `compute_traffic_norm_constants()`
- **Consistency**: Same normalization across all training runs

---

## Adaptive Reward (Not Yet Implemented)

### Motivation
Current reward penalizes extra routers uniformly. Future extension:
- **High traffic** → extra routers help with load distribution
- **Low traffic** → extra routers waste energy

### Proposed Future Modification

```python
def calculate_reward(self, e, q, traffic_intensity):
    # ... existing calculations ...

    # Compute traffic intensity (0 = low, 1 = high)
    # traffic_intensity = total_flows / max_expected_flows

    # Adaptive r penalty
    # High traffic → r_weight ≈ 0.5 → less penalty for extra routers
    # Low traffic → r_weight ≈ 1.0 → full penalty
    r_weight = 1.0 - 0.5 * traffic_intensity

    if not failed:
        reward = np.exp(-e_norm) * np.exp(q) * np.exp((1-r) * r_weight)

    return reward
```

### Intuition
| Traffic Level | r_weight | Effect |
|---------------|----------|--------|
| Low (0.0) | 1.0 | Full penalty for extra routers |
| Medium (0.5) | 0.75 | Moderate penalty |
| High (1.0) | 0.5 | Relaxed penalty (load balancing valuable) |

---

## Training Pipeline

### Current Implementation (Traffic-Aware)
```python
# Extract traffic features grouped by QoS type
flows_by_qos = extract_flows_by_qos(row, client_gateways, server_gateways)

# Get node features with per-QoS flow betweenness centrality
state = get_state(adj_matrix, client_gws, server_gws, original_adj, flows_by_qos)

# Get traffic features (normalized)
traffic_features = get_traffic_features(row, traffic_norm_constants, qos_profiles)

# Agent decision with cross-attention
actions, probs, logits, attn_weights = agent.get_action(state, adj_matrix, traffic_features)

# Attention weights [N × 4] show which QoS types influence each node
```

---

## Implementation Summary

| Component | Base Agent | Traffic-Aware Agent (Current) |
|-----------|------------|-------------------------------|
| **Input** | Graph features (18D) | Graph features (22D) + Traffic features (4×8) |
| **Architecture** | GAT + MLP | GAT + Traffic Encoder + Cross-Attention + MLP |
| **Node Features** | 18 features | 22 features (18 + 4 per-QoS FBC) |
| **Output** | Actions [N] | Actions [N] + Attention weights [N×4] |
| **Interpretability** | Limited | Attention reveals QoS-node relationships |
| **Traffic Awareness** | None | Per-QoS type traffic characteristics |

---

## Design Decisions

1. **Traffic data source**: **MAWI CSV** (`TL_MAWI_balanced.csv`)
   - Real-world traffic patterns from MAWI-WIDE measurements
   - Traffic features extracted per-episode from CSV rows
   - Balanced distribution across traffic levels

2. **Timing**: **Known beforehand**
   - Agent sees upcoming traffic characteristics before topology decisions
   - Realistic for scheduled/predictable traffic scenarios
   - Enables informed, traffic-aware routing decisions

3. **What varies per episode**:
   - ✅ **Number of flows per QoS type** - read from MAWI CSV
   - ✅ **Packet sizes and intervals** - read from MAWI CSV
   - ✅ **Traffic level** - varies across CSV rows
   - ❌ **QoS weights and SLAs** - **Fixed** (constant QoS profiles from `sim/utils.py`)

4. **Normalization**:
   - Traffic features normalized using 99th percentile values
   - Constants pre-computed from full MAWI CSV
   - Stored in `traffic_norm_constants.json` for consistency

---

# Training Infrastructure

## Multi-Run Training System

Training uses `multiple_run.py` to execute consecutive 100-epoch runs:
- Deletes checkpoint at start for fresh training
- Runs `Main.py` repeatedly (default: 60 runs = 6000 epochs)
- Each run loads checkpoint from previous run
- Continuous learning across multiple training sessions

## Checkpoint System

**File**: `agent_weights.pth`

**Contents**:
```python
{
    'agent_state_dict': ...,
    'optimizer_state_dict': ...,
    'epoch': last_epoch + 1,
    'loss_history': [...],
    'qos_history': [...],
    'energy_history': [...],
    'ratio_history': [...],
    'block_avg_loss': [...],      # Averages per 100-epoch block
    'block_fails_count': [...],
    'block_avg_energy': [...],
    'block_avg_qos': [...],
    'block_avg_r': [...]
}
```

## Failure Handling

### SIM FAIL (Simulation Crash)
- **Cause**: Disconnected topology, NS-3 routing failure
- **Behavior**:
  - Prints debug message if occurs at epoch X99
  - Skips metric recording (no data added to block)
  - Advances to next traffic configuration
  - Continues training without disruption

### REAL FAIL (Packet Loss)
- **Cause**: Packets fail to reach destination despite valid paths
- **Behavior**:
  - Records metrics with penalty reward
  - Adds to block averages (included in training)
  - Agent learns to avoid these configurations

### Block Saving Logic

**Normal case** (successful epoch X99):
```python
if (epoch + 1) % 100 == 0 and block_losses:
    # Calculate averages, save block, plot
```

**SIM FAIL case** (failed epoch X99):
```python
# After loop ends:
if block_losses:
    # Save incomplete block with accumulated data
    # Ensures no blocks are lost due to final epoch failure
```

**Key fix**: Blocks are saved even when the last epoch (X99) fails, preventing data loss.

## Plotting & Visualization

**File**: `results.png`

**Format**: 5 subplots tracking metrics over 100-epoch blocks
1. Average Loss (purple)
2. Path Unreachability Count (blue)
3. Average Energy (red)
4. Average QoS (black)
5. Average Router Ratio (green)

**Update frequency**:
- Every 100 epochs (end of each run)
- On interrupted/crashed runs (saves partial progress)

## Debug Output

Training includes comprehensive debug logging:
- `⚠️ DEBUG: SIM FAIL at last epoch of block!` - Critical failure at block boundary
- `✓ DEBUG: Normal plot saved` - Successful block completion
- `✓ DEBUG: Final plot saved` - Incomplete block saved after SIM FAIL
- `✓ DEBUG: Checkpoint saved` - Confirms persistence

---

## Future Extensions

1. **Adaptive reward**: Traffic-intensity-based r penalty (currently uniform)
2. **Temporal modeling**: RNN/Transformer for traffic sequence handling
3. **Multi-step lookahead**: Consider future traffic windows
4. **Hierarchical attention**: Attend to QoS types, then specific flows
5. **Edge-level traffic**: Add traffic as edge features (link utilization)
