import subprocess
import os
import torch
import matplotlib.pyplot as plt
from ns import ns
import random
from utils import *
from agent import Agent
from rl_env import NetworkEnv

os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"

agent = Agent(num_node_features=11, hidden_channels1=64, hidden_channels2=32)
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
original_adj_matrix = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
    [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 22
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 23
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # 24
]
adj_matrix = original_adj_matrix.copy()

n_clients = 1
n_servers = 1

client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)

ip_to_node, node_to_ip = generate_ip_node_mappings(
    original_adj_matrix, n_clients, n_servers
)

loss_history = []
energy_history = []
block_avg_loss = []
block_fails_count = []
block_avg_energy = []

if os.path.exists('./agent_weights.pth'):
    agent.load_state_dict(torch.load(
        './agent_weights.pth', weights_only=True))
for epoch in range(1000):

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
    )

    metrics, reward, fail, ratio, e, q = env.step()

    log_prob = torch.log(p) * actions + torch.log(1-p) * (1-actions)
    entropy = - (p * torch.log(p + 1e-8) + (1 - p)
                 * torch.log(1 - p + 1e-8)).sum()
    entropy_weight = 0.01
    loss = -torch.sum(log_prob * reward) - entropy_weight * entropy
    loss_value = loss.item()

    loss_history.append(loss_value)
    energy_history.append(e)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    if reward == -1 and len(list(nx.all_simple_paths(nx.from_numpy_array(
            np.array(adj_matrix)), client_gateways[0], server_gateways[0]))) > 0:
        print("="*20, " 1FAIL1 ", "="*20)

    print(
        f"Epoch {epoch}, Reward: {reward}, Loss: {loss_value:.4f}, e: {e:.4f}, q: {q}, r: {ratio}")
    print("Sigmoid probabilities:", p.view(-1))
    print("Sampled actions:", actions.view(-1))

    if (epoch + 1) % 100 == 0:
        recent_losses = loss_history[-100:]
        recent_energy = energy_history[-100:]

        avg_loss = sum(recent_losses) / 100.0
        fails = sum(1 for L in recent_losses if L < 0)
        avg_energy = sum(recent_energy) / 100.0

        block_avg_loss.append(avg_loss)
        block_fails_count.append(fails)
        block_avg_energy.append(avg_energy)

        x = [(i+1) * 100 for i in range(len(block_avg_loss))]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
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
        axes[2].set_xlabel('Epochs')
        axes[2].grid(True)

        plt.savefig('results.png')
        plt.close()

    if env is not None:
        ns.Simulator.Destroy()
        env = None
        client_gateways, server_gateways = get_gw(
            adj_matrix, n_clients, n_servers)

torch.save(agent.state_dict(), "./agent_weights.pth")
print('\n\n', '-'*50, ' Saved ', '-'*50, '\n\n')
