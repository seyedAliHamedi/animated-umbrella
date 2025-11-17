# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Reinforcement Learning-based Network Topology Optimization** project that uses Graph Neural Networks (GNN) to optimize network routing decisions in NS-3 simulations. The agent learns to minimize energy consumption while maintaining quality of service (QoS) by selectively enabling/disabling network routers based on real-world MAWI network traffic data.

## Prerequisites and Setup

This project requires NS-3 (Network Simulator 3) to be installed. Follow the installation guide in `README.md` for platform-specific setup instructions (Mac/Linux/Windows WSL).

**Critical Environment Variables** (must be set before running):
```bash
export NS3_HOME=~/ns-3-dev
export NS3_BUILD=$NS3_HOME/build
export PYTHONPATH=$PYTHONPATH:$NS3_BUILD/bindings/python
```

## Running the Project

### Single Training Run
```bash
python Main.py
```

### Multiple Consecutive Runs
```bash
python multiple_run.py  # Runs 60 consecutive training sessions
```

### Important Notes on Execution
- The simulation suppresses C++ warnings via `os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"`
- Training state (including weights, metrics, and epoch number) is saved to `./agent_weights.pth`
- Results visualization is automatically generated as `results.png` every 100 epochs
- The simulator must be destroyed with `ns.Simulator.Destroy()` between episodes to prevent memory leaks

## Architecture Overview

### Core Components

**Main.py** - Main training loop that:
- Initializes the GNN-based agent (GAT architecture with 18 node features)
- Loads MAWI network traffic configurations from CSV (`./TL_MAWI-WIDE_2023-2025.csv` at line 136)
- Manages the RL training loop across 1000+ epochs
- Uses checkpointing to resume training from `./agent_weights.pth`
- Tracks 5 key metrics: loss, path unreachability, energy, QoS, and router utilization ratio

**agent.py** - GNN Agent using PyTorch Geometric:
- Graph Attention Network (GAT) with skip connections
- 18 node features including: router specs, RTT metrics, flow betweenness centrality
- Policy gradient optimization with entropy regularization
- Outputs binary actions (enable/disable) for each router node

**rl_env.py** - NetworkEnv Class:
- Wraps NS-3 simulations as an RL environment
- Manages multiple concurrent applications (clients/servers)
- Computes energy based on router power consumption models (idle/rx/tx states)
- Calculates weighted QoS from flow statistics (delay, jitter, loss, throughput)
- Reward function: `exp(-e_norm) * exp(q) * exp(1-r)` where e=energy, q=qos, r=router ratio

**utils.py** - Graph utilities and metrics:
- Flow Betweenness Centrality (FBC) computation with K-hop path weighting
- RTT feature extraction from pre-computed ping measurements (`ping.csv`)
- Network graph metrics (degree, clustering, articulation points)
- IP-to-node mappings for NS-3 network configuration

### Simulation Infrastructure (sim/ directory)

**topology.py** - Network topology creation:
- Builds NS-3 network from adjacency matrix
- Configures MAWI-WIDE Japanese backbone network (13 nodes representing cities)
- Sets up point-to-point links with specified data rates, delays, queues, and error rates

**app.py** - Application layer:
- Creates client/server pairs connected to gateway nodes
- Supports both UDP and TCP traffic patterns
- Configures traffic based on MAWI dataset parameters (packet size, interval, QoS type)
- Handles dynamic application configuration per flow

**monitor.py** - Metrics collection:
- Flow statistics tracking (packets, delay, jitter, loss)
- Packet-level logging to CSV (`sim/monitor/logs/packets_log.csv`)
- Route tracing to identify active path routers
- NetAnim animation file generation

**sim/utils.py** - Simulation utilities:
- Router specifications with power consumption models (P_idle, P_rx, P_tx, P_base)
- QoS profiles for different traffic types (web, video, gaming, etc.)
- Sample data configurations for topology, links, and applications

## Key Data Files

- `ping.csv` - Pre-computed RTT measurements for 8-node topology (64 pairs, nodes 0-7) - corresponds to commented-out topology on Main.py:97-105
- `ping_MAWI.csv` - Pre-computed RTT measurements for 13-node MAWI topology (169 pairs, nodes 0-12) - **should be used** with current active topology
- `TL_MAWI-WIDE_2023-2025.csv` - Real-world MAWI traffic configurations (loaded at Main.py:136) - contains per-flow throughput, packets, QoS type
- `agent_weights.pth` - Training checkpoint with model state and history
- `results.png` - Visualization of 5 metrics over training blocks

## Training Details

### Node Features (18-dimensional)
1. Is client/server gateway (binary)
2-3. Flow betweenness centrality (original/current topology)
4-8. Router power specs (P_idle, P_rx, P_tx, P_base, queue size) - normalized
9-18. RTT features (neighbor avg/min/max, gateway distances, ratios)

### Reward Function
- **Failure penalty**: `-1 * (n_failed / n_total)` if packets fail to reach destination
- **Success reward**: `exp(-e_norm) * exp(q) * exp(1-r)` where:
  - e_norm: normalized energy consumption (divided by 1717200 for MAWI topology)
  - q: weighted QoS score (0-1, based on SLA compliance)
  - r: ratio of active routers to path routers (penalizes unnecessary active routers)

### Topology Configuration
The current topology is the **MAWI-WIDE Japanese backbone network** with 13 cities:
- 0: Hiroshima, 1: Sakyo, 2: Dojima, 3: Nara, 4: Komatso
- 5: NTT Otemachi, 6: Tsukuba, 7: KDDI Otemachi, 8: Akihabara
- 9: Nezu, 10: Yogami, 11: Hiyoshi, 12: Fujisawa

Adjacency matrix is defined in `Main.py` as `original_adj_matrix`.

## Common Gotchas

1. **NS-3 Simulator must be destroyed** between episodes: Always call `ns.Simulator.Destroy()` and set `env = None` after each simulation run
2. **RTT file mismatch (CRITICAL BUG)**: `utils.py:609` loads `ping.csv` (8 nodes) but the active topology has 13 nodes. Should use `ping_MAWI.csv` instead. This causes nodes 8-12 to get incorrect RTT features (default penalty values of 1000.0/0.5), degrading GNN performance for 5 out of 13 nodes
3. **Zero-packet flows validation**: `process_row()` at Main.py:130 now filters flows where either `/T` (throughput) or `/P` (packets) is zero. This prevents division-by-zero crashes in app.py:178 when calculating packet intervals
4. **MAWI CSV processing**: The `process_row()` function in Main.py re-indexes flow columns to handle sparse flow data (filters based on `/T` column)
5. **Gateway selection**: Client and server gateways are randomly shuffled at the start but paired consistently (client[i] → server[i % n_servers])
6. **Energy normalization**: The constant (1717200) is topology-specific and should be recalculated if topology changes
7. **Gradient clipping**: Agent uses `max_norm=0.5` to prevent exploding gradients
8. **Action override**: Currently actions are zeroed out (`actions = torch.zeros(...)`) on line 183 of Main.py, effectively disabling the learning - this is likely for debugging/baseline comparison
9. **Multiple topology configurations**: Several commented-out topologies exist in Main.py (6-node, 8-node, 9-node, 25-node). The 8-node topology (lines 97-105) corresponds to `ping.csv`, while the active 13-node MAWI topology requires `ping_MAWI.csv`

## Development Notes

- The project uses NetworkX for graph analysis alongside NS-3 for network simulation
- PyTorch Geometric handles GNN operations (GAT layers)
- Matplotlib with 'Agg' backend for headless plotting
- The `sim/monitor/` directory is gitignored as it contains large log files
- Multiple commented-out topology configurations exist in Main.py for experimentation
