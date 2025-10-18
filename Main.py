import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import subprocess
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ns import ns
import random
from utils import *
from agent import Agent
from rl_env import NetworkEnv
import time
import pandas as pd


os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"
t = time.time()
agent = Agent(num_node_features=18, hidden_channels1=64, hidden_channels2=32)
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
    non_fx = ['date', 'time', 'timestamp', 'Total/T', 'Total/P']
    sfxs = ['/T', '/P', '/Avg_packet_size', '/n_packets', '/interval', '/q_type', '_ips']
    v_idx = [int(re.search(r'\d+', c).group()) for c in sorted([c for c in row.index if re.match(r'F\d+/T', c)], key=lambda x: int(re.search(r'\d+', x).group())) if row.get(c, 0) != 0]
    fx_data = {f'F{n_i}{s}': row[f'F{o_i}{s}'] for n_i, o_i in enumerate(v_idx, 1) for s in sfxs if f'F{o_i}{s}' in row}
    return pd.Series({**row[non_fx].to_dict(), **fx_data}), len(v_idx)

adj_matrix = original_adj_matrix.copy()
conf = pd.read_csv("./t/mawi_monthly_csvs/final/MAWI-WIDE_2023-2025.csv")
row=conf.iloc[0]
fx_t_columns = [col for col in conf.columns if col.startswith('F') and col.endswith('/T')]
row, non_zero_count = process_row(conf.iloc[0])
print("/"*20)
print(row)
print("/"*20)
n_clients = non_zero_count
n_servers = non_zero_count

client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)
print(non_zero_count)
print("client gw: ", client_gateways)
print("server gw: ", server_gateways)

ip_to_node, node_to_ip = generate_ip_node_mappings(
    original_adj_matrix, n_clients, n_servers
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
fails = 0
start_epoch = 0

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

for epoch in range(start_epoch, start_epoch + 100):

    print('-'*20, f" Epoch: {epoch} ", '-'*20)

    m = get_state(adj_matrix, client_gateways,
                  server_gateways, original_adj_matrix)
    actions, p, logits = agent.get_action(m, adj_matrix)

    adj_matrix = changeAdj(actions, original_adj_matrix)

    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix,
        original_adj_matrix=original_adj_matrix,
        n_clients=n_clients,
        n_servers=n_servers,
        client_gateways=client_gateways,
        server_gateways=server_gateways,
        ip_to_node=ip_to_node,
        node_to_ip=node_to_ip,
        conf = row,
        n_apps=non_zero_count,
    )

    metrics, reward, fail, ratio, e, q = env.step()
    fails += fail

    log_prob = torch.log(p) * actions + torch.log(1-p) * (1-actions)
    entropy = - (p * torch.log(p + 1e-8) + (1 - p)
                 * torch.log(1 - p + 1e-8)).sum()
    entropy_weight = 0.01
    loss = -torch.sum(log_prob * reward) - entropy_weight * entropy
    loss_value = loss.item()

    loss_history.append(loss_value)
    # print("fail: ", fail)
    # if fail != 0:
    energy_history.append(e)
    qos_history.append(q)
    # print("Q", q)
    # print("E", e)
    if ratio != 0:
        ratio_history.append(ratio)
    if fail and len(list(nx.all_simple_paths(nx.from_numpy_array(
            np.array(adj_matrix)), client_gateways[0], server_gateways[0]))) > 0:
        print("="*20, " 1FAIL1 ", "="*20)
    else:
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    print(
        f"Epoch {epoch}, Reward: {reward}, Loss: {loss_value:.4f}, e: {e:.4f}, q: {q}, r: {ratio}")
    print("Sigmoid probabilities:", p.view(-1))
    print("Sampled actions:", actions.view(-1))

    if (epoch + 1) % 100 == 0:
        recent_losses = loss_history[-100:]
        recent_energy = energy_history[-100:]
        recent_qos = qos_history[-100:]
        recent_ratios = ratio_history[-100:]

        avg_loss = sum(recent_losses) / 100.0
        # fails = sum(1 for L in recent_losses if L < 0)

        avg_energy = sum(recent_energy) / 100.0
        avg_qos = sum(recent_qos) / 100.0
        avg_r = sum(recent_ratios) / 100.0

        block_avg_loss.append(avg_loss)
        block_fails_count.append(fails)
        fails = 0
        block_avg_energy.append(avg_energy)
        block_avg_qos.append(avg_qos)
        block_avg_r.append(avg_r)

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

    if env is not None:
        ns.Simulator.Destroy()
        env = None
        adj_matrix = original_adj_matrix.copy()
        row=conf.iloc[epoch+1]   

        non_zero_count = sum(1 for col in fx_t_columns if row1[col] != 0)
        n_clients = non_zero_count
        n_servers = non_zero_count

        client_gateways, server_gateways = get_gw(
            adj_matrix, n_clients, n_servers)

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
    'block_avg_r': block_avg_r
}, "./agent_weights.pth")
print('\n\n', '-'*50, ' Saved ', '-'*50, '\n\n')
print("HEHEHEHHEHEHEHEH", time.time()-t)
