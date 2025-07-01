from ns import ns

import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *
from agent import Agent
from rl_env import NetworkEnv
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


agent = Agent(num_node_features=11, hidden_channels1=32,
              hidden_channels2=64, lr=0.0005)
ppo_epochs = 10
ppo_eps = 0.2
ppo_batch = 10
entropy_start = 0.05
entropy_end = 0.01
entropy_decay = 500


original_adj_matrix = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
]
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

adj_matrix = original_adj_matrix.copy()
n_clients = 1
n_servers = 1
# client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)

ip_to_node, node_to_ip = generate_ip_node_mappings(
    original_adj_matrix, n_clients, n_servers)

reward_history = []
energy_history = []
ratio_history = []
block_avg_reward = []
block_fails_count = []
block_avg_energy = []
block_avg_ratio = []

if os.path.exists('./agent_weights.pth'):
    agent.load_state_dict(torch.load(
        './agent_weights.pth', weights_only=True))

for epoch in range(1000):
    print('-'*50)
    print('-'*20, f" Epoch: {epoch} ", '-'*20)
    print('-'*50, '\n')

    batch_states = []
    batch_adj = []
    batch_actions = []
    batch_old_log_probs = []
    batch_rewards = []
    agent.eval()
    with torch.no_grad():
        for batch in range(ppo_batch):
            client_gateways, server_gateways = get_gw(
                adj_matrix, n_clients, n_servers)
            m = get_state(adj_matrix, client_gateways,
                          server_gateways, original_adj_matrix)

            batch_states.append(m)
            batch_adj.append(adj_matrix)

            logits, _ = agent(agent.dict_to_data(adj_matrix, m))
            p = torch.sigmoid(logits)
            actions = torch.bernoulli(p)

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

            log_prob = - F.binary_cross_entropy_with_logits(
                logits, actions, reduction='sum')

            batch_rewards.append(reward)
            batch_actions.append(actions)
            batch_old_log_probs.append(log_prob)

            if reward == -1 and len(list(nx.all_simple_paths(nx.from_numpy_array(np.array(adj_matrix)), client_gateways[0], server_gateways[0]))) > 0:
                print("="*20, " 1FAIL1 ", "="*20)

            print("Sigmoid probabilities:", [
                  f"{x:.4f}" for x in p.view(-1).detach().cpu().numpy()])

            reward_history.append(reward)
            energy_history.append(e)
            ratio_history.append(ratio)

            if env is not None:
                ns.Simulator.Destroy()
            env = None

        old_log_probs_tensor = torch.stack(batch_old_log_probs)
        batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_actions_tensor = torch.stack(batch_actions)

    agent.train()
    for ppo_epoch in range(ppo_epochs):
        values = []
        new_log_probs = []
        entropies = []
        for batch in range(ppo_batch):
            m = batch_states[batch]
            adj = batch_adj[batch]
            new_logits, value = agent(agent.dict_to_data(adj, m))
            values.append(value)

            p = torch.sigmoid(new_logits)
            entropy = -(p * torch.log(p + 1e-9) + (1 - p)
                        * torch.log(1 - p + 1e-9)).sum()

            entropies.append(entropy)

            old_actions = batch_actions_tensor[batch]
            new_log_prob = - \
                F.binary_cross_entropy_with_logits(
                    new_logits, old_actions, reduction='sum')

            new_log_probs.append(new_log_prob)

        values_tensor = torch.stack(values)
        new_log_prob_tensor = torch.stack(new_log_probs)
        entropy_bonus = torch.stack(entropies).mean()

        # returns = torch.stack(compute_gae(
        # batch_rewards_tensor, values_tensor.detach(), values_tensor[-1].detach()))
        advantages = batch_rewards_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        ratio = torch.exp(new_log_prob_tensor - old_log_probs_tensor)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(
            batch_rewards_tensor.squeeze(-1), values_tensor)

        entropy_weight = entropy_end + \
            (entropy_start - entropy_end) * \
            math.exp(-1. * epoch / entropy_decay)

        loss = actor_loss + 0.5 * critic_loss - entropy_weight*entropy_bonus

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    print("EPOCH mean reward", batch_rewards_tensor.mean(axis=0))

    if (epoch + 1) % 100 == 0 and epoch > 0:
        recent_rewards = reward_history[-100:]
        recent_energy = energy_history[-100:]
        recent_ratio = ratio_history[-100:]

        avg_reward = sum(recent_rewards) / len(recent_rewards)
        fails = sum(1 for r in recent_rewards if r < 0) / len(recent_rewards)
        avg_energy = sum(recent_energy) / len(recent_energy)
        avg_ratio = sum(recent_ratio) / len(recent_ratio)

        block_avg_reward.append(avg_reward)
        block_fails_count.append(fails)
        block_avg_energy.append(avg_energy)
        block_avg_ratio.append(avg_ratio)

        x = [(i+1) * 100 for i in range(len(block_avg_reward))]

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(
            f'Metrics by 100-Epoch Block up to Epoch {epoch+1}', fontsize=16)

        axes[0].plot(x, block_avg_reward, color='green',
                     marker='o', label='Avg Reward')
        axes[0].set_ylabel('Avg Reward')
        axes[0].grid(True)

        axes[1].plot(x, block_fails_count, color='blue',
                     marker='o', label='Fail Rate')
        axes[1].set_ylabel('Avg Fail Rate')
        axes[1].grid(True)

        axes[2].plot(x, block_avg_energy, color='red',
                     marker='o', label='Avg Energy')
        axes[2].set_ylabel('Avg Energy (e)')
        axes[2].grid(True)

        axes[3].plot(x, block_avg_ratio, color='purple',
                     marker='o', label='Avg Ratio')
        axes[3].set_ylabel('Avg Ratio (Active/Needed)')
        axes[3].set_xlabel('Epochs')
        axes[3].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('results.png')
        plt.close()

    if epoch and epoch % 100 == 0:
        torch.save(agent.state_dict(), "./agent_weights.pth")
print('\n\n', '-'*50, ' Saved ', '-'*50, '\n\n')
