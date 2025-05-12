import os
import torch
import matplotlib.pyplot as plt
from ns import ns
import random
from utils import *
from agent import Agent
from rl_env import NetworkEnv

# suppress CPPYY warnings
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"

# initialize agent
agent = Agent(num_node_features=11, hidden_channels1=32, hidden_channels2=16)
torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

# original graph
# original_adj_matrix = [
#     [0, 0, 0, 1],
#     [0, 0, 0, 1],
#     [0, 0, 0, 1],
#     [1, 1, 1, 0]
# ]
original_adj_matrix = [
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0]
]
adj_matrix = original_adj_matrix.copy()

n_clients = 1
n_servers = 1

# initial gateway assignment
client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)

# IP/node mappings cache
ip_to_node, node_to_ip = generate_ip_node_mappings(
    original_adj_matrix, n_clients, n_servers
)

# histories for raw per-epoch metrics
loss_history = []
energy_history = []

# histories for block-wise (every 100 epochs) metrics
block_avg_loss = []
block_fails_count = []
block_avg_energy = []

# ensure plot directory exists
os.makedirs('plots', exist_ok=True)

for epoch in range(20000):
    print('-'*20, f" Epoch: {epoch} ", '-'*20)

    # get state and select action
    m = get_state(adj_matrix, client_gateways,
                  server_gateways, original_adj_matrix)
    actions, p, logits = agent.get_action(m, adj_matrix)

    # apply action to adjacency
    adj_matrix = changeAdj(actions, original_adj_matrix)

    # create environment with new graph
    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix,
        original_adj_matrix=original_adj_matrix,
        n_clients=1,
        n_servers=1,
        client_gateways=client_gateways,
        server_gateways=server_gateways,
        ip_to_node=ip_to_node,
        node_to_ip=node_to_ip,
    )

    # run one step
    metrics, reward, e, q = env.step()
    log_prob = torch.log(p) * actions + torch.log(1-p) * (1-actions)
    loss = -torch.sum(log_prob * reward)
    loss_value = loss.item()

    # record per-epoch metrics
    loss_history.append(loss_value)
    energy_history.append(e)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # logging
    print(
        f"Epoch {epoch}, Reward: {reward}, Loss: {loss_value:.4f}, e: {e:.4f}, q: {q}")
    print("Sigmoid probabilities:", p.view(-1))
    print("Sampled actions:", actions.view(-1))

    # every 100 epochs, compute block metrics & plot
    if (epoch + 1) % 100 == 0:
        # get last 100 values
        recent_losses = loss_history[-100:]
        recent_energy = energy_history[-100:]

        # block metrics
        avg_loss = sum(recent_losses) / 100.0
        fails = sum(1 for L in recent_losses if L < 0)
        avg_energy = sum(recent_energy) / 100.0

        block_avg_loss.append(avg_loss)
        block_fails_count.append(fails)
        block_avg_energy.append(avg_energy)

        # prepare x-axis (epoch boundaries)
        x = [(i+1) * 100 for i in range(len(block_avg_loss))]

        # plot
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

        # save and close
        plot_path = f'results.png'
        fig.savefig(plot_path)
        plt.close(fig)

    # teardown ns simulation and recompute gateways
    if env is not None:
        ns.Simulator.Destroy()
        env = None
        client_gateways, server_gateways = get_gw(
            adj_matrix, n_clients, n_servers
        )
