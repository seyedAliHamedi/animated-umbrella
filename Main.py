import os
import torch
import subprocess

from utils import *
import middle
from agent import Agent
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"

n = 15
# Baseline connectivity when all nodes are active.
original_adj_matrix = fc_graph(n)
adj_matrix = original_adj_matrix.copy()
middle.data['adj_matrix'] = adj_matrix
middle.save_data()
agent = Agent(num_node_features=24, hidden_channels1=64, hidden_channels2=32)

for epoch in range(20):

    subprocess.run(["python", "single.py", str(epoch)])
    middle.load_data()
    metric = middle.data['metrics']
    actions, p, logits = agent.get_action(metric, adj_matrix)
    adj_matrix = changeAdj(actions, original_adj_matrix)
    middle.data['adj_matrix'] = adj_matrix
    middle.save_data()
    reward = middle.data['reward']
    print("Reward: ", reward)

    # Calculate the log probability for each node action.
    # actions is 1 for turning off and 0 for staying on.
    log_prob = actions * torch.log(p) + (1 - actions) * torch.log(1 - p)

    # Multiply the log probability by the reward (assuming reward is a scalar or a broadcastable tensor).
    loss = -torch.sum(log_prob) * reward

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    print("-" * 20)
    print(f"LOSS: {loss.item()}")
    print(f"Updated Adjacency Matrix: {adj_matrix}")

print("All epochs completed")
