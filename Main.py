# ============================================================
# CONFIGURATION FLAGS (must be set BEFORE other imports)
# ============================================================
import os
import warnings

SHOW_TIMING = False
FILTER_TRAFFIC_LEVELS = False  # Enable/disable traffic level filtering
TRAFFIC_LEVELS = [4]  # Configure which traffic levels to simulate (only used if FILTER_TRAFFIC_LEVELS is True)
# ============================================================

os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"
os.environ["SHOW_TIMING"] = str(int(SHOW_TIMING))  # Export for other modules

# Now safe to import modules that depend on environment variables
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import time
from rl_env import NetworkEnv
from agent import Agent, TrafficAwareAgent
from utils import (
    get_state, changeAdj, get_gw, generate_ip_node_mappings,
    extract_flows_by_qos, get_traffic_features, compute_traffic_intensity,
    load_traffic_norm_constants, QOS_TYPES
)
from sim.utils import sample_data as sim_sample_data  # For QoS profiles
import random
from ns import ns
import matplotlib.pyplot as plt
import matplotlib
import torch
import subprocess
import networkx as nx
import numpy as np
import re

matplotlib.use('Agg')
t = time.time()


USE_TRAFFIC_AWARE = True  # Set to False to use original Agent

# 22 features: 18 original + 4 per-QoS FBC
if USE_TRAFFIC_AWARE:
    agent = TrafficAwareAgent(num_node_features=22, hidden=64, n_qos_types=4, traffic_feat_per_qos=8)
else:
    agent = Agent(num_node_features=22, hidden_channels1=64, hidden_channels2=32)
torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

# original_adj_matrix = [
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [1, 1, 0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 1, 1, 1, 0, 0, 1, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
# ]
# original_adj_matrix = [
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
#     [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
#     [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
#     [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
#         0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
#     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 12
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
#         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 15
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 16
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 17
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 18
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 19
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 20
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 21
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 22
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 23
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # 24
# ]

# original_adj_matrix = [
#     [0, 1, 0, 0, 1, 0],
#     [1, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 1],
#     [0, 0, 1, 0, 0, 1],
#     [1, 1, 0, 0, 0, 1],
#     [0, 0, 1, 1, 1, 0],
# ]
# original_adj_matrix = [
#     [0, 1, 0, 0, 1, 0, 1, 0],
#     [1, 0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 1, 0, 0],
#     [1, 0, 0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 1, 1, 0, 0, 1],
#     [1, 1, 0, 0, 1, 0, 0, 1],
#     [0, 0, 1, 1, 0, 1, 1, 0],
# ] #latest

original_adj_matrix = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 Hiroshima
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 Sakyo
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],  # 2 Dojima
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 3 Nara
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 4 Komatso
    [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],  # 5 NTT Otemachi
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 6 Tsukuba
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # 7 KDDI Otemachi
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 8 Akihabara
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 9 Nezu
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],  # 10 Yogami
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 11 Hiyoshi
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 12 Fujisawa
]


def process_row(row):
    non_fx = ['date', 'time', 'timestamp', 'Total/T', 'Total/P', 'traffic_level']
    sfxs = ['/T', '/P', '/Avg_packet_size',
            '/n_packets', '/interval', '/q_type', '_ips']

    # Filter flows: exclude if T or P is NaN or zero
    v_idx = [int(re.search(r'\d+', c).group()) for c in sorted([c for c in row.index if re.match(
        r'F\d+/T', c)], key=lambda x: int(re.search(r'\d+', x).group()))
        if not pd.isna(row.get(c, 0)) and not pd.isna(row.get(c.replace('/T', '/P'), 0))
        and row.get(c, 0) != 0 and row.get(c.replace('/T', '/P'), 0) != 0]

    fx_data = {f'F{n_i}{s}': row[f'F{o_i}{s}'] for n_i, o_i in enumerate(
        v_idx, 1) for s in sfxs if f'F{o_i}{s}' in row}
    return pd.Series({**row[non_fx].to_dict(), **fx_data}), len(v_idx)


adj_matrix = original_adj_matrix.copy()
# conf = pd.read_csv("./timestamps/TL_MAWI-WIDE_2023-2025.csv")
CSV_PATH = "./timestamps/TL_MAWI_balanced.csv"
conf = pd.read_csv(CSV_PATH)
# Filter to only include rows with specified traffic levels (if enabled)
if FILTER_TRAFFIC_LEVELS:
    conf = conf[conf['traffic_level'].isin(TRAFFIC_LEVELS)].reset_index(drop=True)
    print(f"Filtered dataset to {len(conf)} rows with traffic levels: {TRAFFIC_LEVELS}")
else:
    print(f"Using full dataset with {len(conf)} rows (no traffic level filtering)")

# Load traffic normalization constants (pre-computed from main CSV)
if USE_TRAFFIC_AWARE:
    import json
    try:
        print("Loading traffic normalization constants from traffic_norm_constants.json...")
        traffic_norm_constants = load_traffic_norm_constants('./traffic_norm_constants.json')
        print(f"✓ Loaded norm constants for {len(QOS_TYPES)} QoS types")
    except FileNotFoundError:
        print("traffic_norm_constants.json not found!")
        print("Computing normalization constants from CSV (this may take a moment)...")
        from utils import compute_traffic_norm_constants
        traffic_norm_constants = compute_traffic_norm_constants(CSV_PATH, percentile=99)

        # Save the generated constants for future runs
        with open('./traffic_norm_constants.json', 'w') as f:
            json.dump(traffic_norm_constants, f, indent=2)
        print(f"✓ Generated and saved traffic_norm_constants.json")
        print(f"✓ Computed norm constants for {len(QOS_TYPES)} QoS types")

    qos_profiles = sim_sample_data['mawi_q_list']
row = conf.iloc[0]
fx_t_columns = [col for col in conf.columns if col.startswith(
    'F') and col.endswith('/T')]
row, non_zero_count = process_row(conf.iloc[0])
n_clients = non_zero_count
n_servers = non_zero_count

client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)

# print("client gw: ", client_gateways)
# print("server gw: ", server_gateways)

ip_to_node, node_to_ip = generate_ip_node_mappings(
    original_adj_matrix, len(adj_matrix), len(adj_matrix)
)

loss_history = []
qos_history = []
energy_history = []
ratio_history = []
block_avg_loss = []
block_fails_count = []
block_avg_energy = []
block_avg_qos = []
block_avg_r = []
block_losses = []
block_energies = []
block_qos = []
block_ratios = []
successful_epochs_in_block = 0
fails = 0
start_epoch = 0

# Per-q_type QoS tracking
QOS_TYPES_LIST = ["Interactive_Web", "Streaming_Media", "Background_Sync", "Real_Time_Interactive"]
block_qos_by_type = {q_type: [] for q_type in QOS_TYPES_LIST}  # Accumulate per epoch within block
block_avg_qos_by_type = {q_type: [] for q_type in QOS_TYPES_LIST}  # Store block averages

if os.path.exists('./agent_weights.pth'):
    checkpoint = torch.load('./agent_weights.pth', weights_only=True)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    qos_history = checkpoint['qos_history']
    energy_history = checkpoint['energy_history']
    ratio_history = checkpoint['ratio_history']
    block_avg_loss = checkpoint['block_avg_loss']
    block_fails_count = checkpoint['block_fails_count']
    block_avg_energy = checkpoint['block_avg_energy']
    block_avg_qos = checkpoint['block_avg_qos']
    block_avg_r = checkpoint['block_avg_r']
    # Load per-q_type tracking if available (for backward compatibility)
    if 'block_avg_qos_by_type' in checkpoint:
        block_avg_qos_by_type = checkpoint['block_avg_qos_by_type']


SIMULATION_TIME = 1
for epoch in range(start_epoch, start_epoch + 100):

    print('-'*20, f" Epoch: {epoch} ", '-'*20)

    # Extract flows grouped by QoS type for per-QoS FBC computation
    if USE_TRAFFIC_AWARE:
        flows_by_qos = extract_flows_by_qos(row, client_gateways, server_gateways)
    else:
        flows_by_qos = None

    m = get_state(adj_matrix, client_gateways,
                  server_gateways, original_adj_matrix, flows_by_qos=flows_by_qos)

    # Get actions from agent (with or without traffic features)
    if USE_TRAFFIC_AWARE:
        traffic_features = get_traffic_features(row, traffic_norm_constants, qos_profiles)
        actions, p, logits, attn_weights = agent.get_action(m, adj_matrix, traffic_features)
    else:
        actions, p, logits = agent.get_action(m, adj_matrix)

    adj_matrix = changeAdj(actions, original_adj_matrix)

    # Compute traffic intensity for adaptive reward
    # Note: traffic_intensity for adaptive reward not implemented yet
    # if USE_TRAFFIC_AWARE:
    #     traffic_intensity = compute_traffic_intensity(row, traffic_norm_constants)

    env = NetworkEnv(
        simulation_duration=SIMULATION_TIME,
        adj_matrix=adj_matrix,
        original_adj_matrix=original_adj_matrix,
        n_clients=n_clients,
        n_servers=n_servers,
        client_gateways=client_gateways,
        server_gateways=server_gateways,
        ip_to_node=ip_to_node,
        node_to_ip=node_to_ip,
        conf=row,
        n_apps=non_zero_count,
    )

    metrics, reward, fail, ratio, e, q, qos_by_type = env.step()

    # If there's a real fail, set QoS to 0 for q_types that completely failed (had flows but no successful packets)
    if fail:
        # Extract expected q_types from the configuration (flows that were supposed to exist)
        if USE_TRAFFIC_AWARE and flows_by_qos:
            expected_q_types = [q_type for q_type, flow_list in flows_by_qos.items() if flow_list]
        else:
            # Fallback: extract from row configuration directly
            flow_cols = [c for c in row.index if c.startswith('F') and c.endswith('/q_type')]
            expected_q_types = list(set([row[c] for c in flow_cols if c in row.index and row[c] in QOS_TYPES_LIST]))

        # For q_types that were expected but not in qos_by_type (completely failed), set to 0
        for q_type in expected_q_types:
            if q_type not in qos_by_type:
                qos_by_type[q_type] = 0.0

    if fail and len(list(nx.all_simple_paths(nx.from_numpy_array(
            np.array(adj_matrix)), client_gateways[0], server_gateways[0]))) > 0:
        print("="*20, f" SIM FAIL TL: {row.get('traffic_level', 'N/A')} ", "="*20)
        ns.Simulator.Destroy()

        # Advance to next row to avoid infinite loop
        adj_matrix = original_adj_matrix.copy()
        next_row_idx = (epoch + 1) % len(conf)
        row, non_zero_count = process_row(conf.iloc[next_row_idx])
        while non_zero_count == 0:
            next_row_idx = (next_row_idx + 1) % len(conf)
            print("Redundant row")
            row, non_zero_count = process_row(conf.iloc[next_row_idx])

        # Update clients/servers for new row
        n_clients = non_zero_count
        n_servers = non_zero_count
        client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)

        continue
    elif fail:
        print("REAL FAIL")

    fails += fail

    log_prob = torch.log(p) * actions + torch.log(1-p) * (1-actions)
    entropy = - (p * torch.log(p + 1e-8) + (1 - p)
                 * torch.log(1 - p + 1e-8)).sum()
    entropy_weight = 0.01
    loss = -torch.sum(log_prob * reward) - entropy_weight * entropy
    loss_value = loss.item()

    loss_history.append(loss_value)
    block_losses.append(loss_value)

    energy_history.append(e)
    block_energies.append(e)

    qos_history.append(q)
    block_qos.append(q)

    # Accumulate per-q_type QoS scores
    for q_type in QOS_TYPES_LIST:
        if q_type in qos_by_type:
            block_qos_by_type[q_type].append(qos_by_type[q_type])

    if ratio != 0:
        ratio_history.append(ratio)
        block_ratios.append(ratio)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    successful_epochs_in_block += 1
    print(
        f"Epoch {epoch}, Reward: {reward}, Loss: {loss_value:.4f}, e: {e:.4f}, q: {q}, r: {ratio}, f: {int(fail)}, TL: {row.get('traffic_level', 'N/A')}")
    # print("Sigmoid probabilities:", p.view(-1))
    # print("Sampled actions:", actions.view(-1))

    # Save plot after every 100 epochs (regardless of failures)
    if (epoch + 1) % 100 == 0 and block_losses:
        avg_loss = sum(block_losses) / len(block_losses)
        avg_energy = sum(block_energies) / len(block_energies)
        avg_qos = sum(block_qos) / len(block_qos)
        avg_r = sum(block_ratios) / len(block_ratios) if block_ratios else 0

        block_avg_loss.append(avg_loss)
        block_fails_count.append(fails)
        block_avg_energy.append(avg_energy)
        block_avg_qos.append(avg_qos)
        block_avg_r.append(avg_r)

        # Compute per-q_type averages for this block
        for q_type in QOS_TYPES_LIST:
            if block_qos_by_type[q_type]:
                avg_qos_for_type = sum(block_qos_by_type[q_type]) / len(block_qos_by_type[q_type])
                block_avg_qos_by_type[q_type].append(avg_qos_for_type)
            else:
                # If no data for this q_type in this block, append None or 0
                block_avg_qos_by_type[q_type].append(None)

        x = [(i+1) * 100 for i in range(len(block_avg_loss))]

        fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
        fig.suptitle(
            f'Metrics by 100-Epoch Block up to Epoch {epoch+1}', fontsize=14)

        axes[0].plot(x, block_avg_loss, color='purple', marker='o')
        axes[0].set_ylabel('Avg Loss History')
        axes[0].grid(True)

        axes[1].plot(x, block_fails_count, color='blue', marker='o')
        axes[1].set_ylabel('Avg Path Unreachability History')
        axes[1].grid(True)

        axes[2].plot(x, block_avg_energy, color='red', marker='o')
        axes[2].set_ylabel('Avg Energy History')
        axes[2].grid(True)

        axes[3].plot(x, block_avg_qos, color='black', marker='o')
        axes[3].set_ylabel('Avg Qos History')
        axes[3].grid(True)

        axes[4].plot(x, block_avg_r, color='green', marker='o')
        axes[4].set_ylabel('Avg Ratio (r) History')
        axes[4].set_xlabel('Epochs')
        axes[4].grid(True)

        plt.savefig('results.png')
        plt.close()

        # Create per-q_type QoS plot
        fig_qos, ax_qos = plt.subplots(figsize=(10, 6))
        fig_qos.suptitle(f'QoS Scores by Type (100-Epoch Blocks) up to Epoch {epoch+1}', fontsize=14)

        for q_type in QOS_TYPES_LIST:
            # Filter out None values and their corresponding x values
            data = [(x_val, y_val) for x_val, y_val in zip(x, block_avg_qos_by_type[q_type]) if y_val is not None]
            if data:
                x_filtered, y_filtered = zip(*data)
                ax_qos.plot(x_filtered, y_filtered, marker='o', label=q_type, linewidth=2)

        ax_qos.set_xlabel('Epochs')
        ax_qos.set_ylabel('Average QoS Score')
        ax_qos.legend(loc='best')
        ax_qos.grid(True)
        plt.tight_layout()
        plt.savefig('QoS.png')
        plt.close()

        block_losses = []
        block_energies = []
        block_qos = []
        block_ratios = []
        # Clear per-q_type accumulators
        block_qos_by_type = {q_type: [] for q_type in QOS_TYPES_LIST}
        successful_epochs_in_block = 0
        fails = 0

    if env is not None:
        ns.Simulator.Destroy()
        env = None
        adj_matrix = original_adj_matrix.copy()
        # Use modulo to cycle through filtered dataset
        next_row_idx = (epoch + 1) % len(conf)
        row, non_zero_count = process_row(conf.iloc[next_row_idx])
        while non_zero_count == 0:
            next_row_idx = (next_row_idx + 1) % len(conf)
            print("Redundant row")
            row, non_zero_count = process_row(conf.iloc[next_row_idx])

        n_clients = non_zero_count
        n_servers = non_zero_count

        client_gateways, server_gateways = get_gw(
            adj_matrix, n_clients, n_servers)

# Save final plot if there's incomplete block data (for interrupted runs)
if block_losses:
    # First, save the current incomplete block before plotting
    avg_loss = sum(block_losses) / len(block_losses)
    avg_energy = sum(block_energies) / len(block_energies)
    avg_qos = sum(block_qos) / len(block_qos)
    avg_r = sum(block_ratios) / len(block_ratios) if block_ratios else 0

    block_avg_loss.append(avg_loss)
    block_fails_count.append(fails)
    block_avg_energy.append(avg_energy)
    block_avg_qos.append(avg_qos)
    block_avg_r.append(avg_r)

    # Compute per-q_type averages for final incomplete block
    for q_type in QOS_TYPES_LIST:
        if block_qos_by_type[q_type]:
            avg_qos_for_type = sum(block_qos_by_type[q_type]) / len(block_qos_by_type[q_type])
            block_avg_qos_by_type[q_type].append(avg_qos_for_type)
        else:
            block_avg_qos_by_type[q_type].append(None)

    x = [(i+1) * 100 for i in range(len(block_avg_loss))]

    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f'Metrics by 100-Epoch Block up to Epoch {epoch+1}', fontsize=14)

    axes[0].plot(x, block_avg_loss, color='purple', marker='o')
    axes[0].set_ylabel('Avg Loss History')
    axes[0].grid(True)

    axes[1].plot(x, block_fails_count, color='blue', marker='o')
    axes[1].set_ylabel('Avg Path Unreachability History')
    axes[1].grid(True)

    axes[2].plot(x, block_avg_energy, color='red', marker='o')
    axes[2].set_ylabel('Avg Energy History')
    axes[2].grid(True)

    axes[3].plot(x, block_avg_qos, color='black', marker='o')
    axes[3].set_ylabel('Avg Qos History')
    axes[3].grid(True)

    axes[4].plot(x, block_avg_r, color='green', marker='o')
    axes[4].set_ylabel('Avg Ratio (r) History')
    axes[4].set_xlabel('Epochs')
    axes[4].grid(True)

    plt.savefig('results.png')
    plt.close()
    print("✓ Saved results.png")

    # Create per-q_type QoS plot for final block
    fig_qos, ax_qos = plt.subplots(figsize=(10, 6))
    fig_qos.suptitle(f'QoS Scores by Type (100-Epoch Blocks) up to Epoch {epoch+1}', fontsize=14)

    for q_type in QOS_TYPES_LIST:
        # Filter out None values and their corresponding x values
        data = [(x_val, y_val) for x_val, y_val in zip(x, block_avg_qos_by_type[q_type]) if y_val is not None]
        if data:
            x_filtered, y_filtered = zip(*data)
            ax_qos.plot(x_filtered, y_filtered, marker='o', label=q_type, linewidth=2)

    ax_qos.set_xlabel('Epochs')
    ax_qos.set_ylabel('Average QoS Score')
    ax_qos.legend(loc='best')
    ax_qos.grid(True)
    plt.tight_layout()
    plt.savefig('QoS.png')
    plt.close()
    print("✓ Saved QoS.png")

torch.save({
    'agent_state_dict': agent.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'epoch': epoch + 1,
    'loss_history': loss_history,
    'qos_history': qos_history,
    'energy_history': energy_history,
    'ratio_history': ratio_history,
    'block_avg_loss': block_avg_loss,
    'block_fails_count': block_fails_count,
    'block_avg_energy': block_avg_energy,
    'block_avg_qos': block_avg_qos,
    'block_avg_r': block_avg_r,
    'block_avg_qos_by_type': block_avg_qos_by_type
}, "./agent_weights.pth")
# print('\n\n', '-'*50, ' Saved ', '-'*50, '\n\n')
print("HEHEHEHHEHEHEHEH", time.time()-t)
