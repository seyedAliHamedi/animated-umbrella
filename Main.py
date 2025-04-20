import os
import torch
from ns import ns
from rl_env import NetworkEnv
from utils import *
from agent import Agent
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


n = 4  # Number of nodes in the network

original_adj_matrix = fc_graph(n)
adj_matrix = original_adj_matrix.copy()

agent = Agent(num_node_features=29, node_hidden_channels=64,
              num_edge_features=11, edge_hidden_channels=64, num_nodes=n, lr=0.001)

edges_index = []
for i in range(n):
    for j in range(i+1, n):
        edges_index.append((i, j))

env = None
prev_actions = None
prev_p = None

for epoch in range(2000):
    if env is not None:
        ns.Simulator.Destroy()
        env = None

    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix,
        original_adj_matrix=original_adj_matrix,
        n_clients=1,
        n_servers=1
    )
    print(f"-------------------Epoch {epoch}-----------------------")

    node_metrics, edge_metrics, reward = env.step()

    if prev_actions is not None:
        log_prob = prev_actions * \
            torch.log(prev_p + 1e-10) + (1 - prev_actions) * \
            torch.log(1 - prev_p + 1e-10)
        loss = -torch.sum(log_prob) * reward

        if reward < -0.01 and torch.all(prev_actions == 0):
            print("Skipping training due to invalid reward on fully active graph")
        else:
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            print(f"Epoch {epoch}, Reward: {reward}, Loss: {loss.item()}")
    else:
        print(f"Epoch {epoch}, First epoch – no previous actions.")

    # Get actions from the agent (now with fixed size output)
    actions, p, edge_index = agent.get_action(
        node_metrics, adj_matrix, edge_metrics)

    print("Sigmoid probabilities:", p)
    print("Sampled actions:", actions)
    for idx, action in enumerate(actions):
        i, j = edges_index[idx]
        print(i, "->", j, ' : ', action)

    # Apply the actions to change the adjacency matrix
    adj_matrix = changeAdj(actions, original_adj_matrix, edge_index)
    prev_actions = actions
    prev_p = p
