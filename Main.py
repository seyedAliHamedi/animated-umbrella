import os
import torch
import subprocess
import time

from ns import ns
from rl_env import NetworkEnv
from utils import *
import middle
from agent import Agent
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


t1 = time.time()
n = 15

original_adj_matrix = fc_graph(n)
adj_matrix = original_adj_matrix.copy()

middle.data['adj_matrix'] = adj_matrix

middle.save_data()
agent = Agent(num_node_features=24, hidden_channels1=64, hidden_channels2=32)


t2 = time.time()
env = None
for epoch in range(2000):

    print(f"Running Epoch {epoch}")
    middle.load_data()
    adj_matrix = middle.data['adj_matrix']
    if env is not None:
        ns.Simulator.Destroy()
        env = None
    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix
    )
    metrics, reward = env.step()
    middle.data['metrics'] = metrics
    middle.data['reward'] = reward
    middle.save_data()
    print(f"    Reward: {reward}")
    print(f"Epoch {epoch} completed")
    print("RUN FILE", time.time()-t2)
    t3 = time.time()
    middle.load_data()
    metric = middle.data['metrics']
    print("LOAD METRICS", time.time()-t3)
    print(metric)
    t4 = time.time()
    actions, p, logits = agent.get_action(metric, adj_matrix)
    print("AGENT", time.time()-t4)

    t5 = time.time()

    adj_matrix = changeAdj(actions, original_adj_matrix)
    middle.data['adj_matrix'] = adj_matrix
    middle.save_data()
    reward = middle.data['reward']
    print("Reward: ", reward)
    print("KOS SHER ", time.time()-t5)

    t6 = time.time()

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
    print(time.time()-t6)

print("All epochs completed")
