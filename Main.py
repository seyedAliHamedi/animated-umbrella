import os
import torch
import subprocess
import time

from ns import ns
from rl_env import NetworkEnv
from utils import *
from agent import Agent
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


n = 4

original_adj_matrix = fc_graph(n)
adj_matrix = original_adj_matrix.copy()

agent = Agent(num_node_features=24, hidden_channels1=64, hidden_channels2=32)
env = None
for epoch in range(2000):
    # print('-'*30, "Running Epoch: ", epoch, '-'*30)

    if env is not None:
        ns.Simulator.Destroy()
        env = None

    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix,
        n_clients=1,
        n_servers=1,
    )

    metrics, reward = env.step()

    actions, p, logits = agent.get_action(metrics, adj_matrix)
    print("Sigmoid probabilities:", p)
    print("Sampled actions:", actions)

    adj_matrix = changeAdj(actions, original_adj_matrix)

    log_prob = actions * torch.log(p + 1e-10) + \
        (1 - actions) * torch.log(1 - p + 1e-10)

    loss = -torch.sum(log_prob) * reward

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    print(f"Epoch {epoch}, Reward: {reward}, Loss: {loss.item()}")
