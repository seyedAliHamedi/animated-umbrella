import os
import time
import torch
import matplotlib.pyplot as plt
from ns import ns
import random
from utils import *
from agent import Agent
from rl_env import NetworkEnv

os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"
agent = Agent(num_node_features=1, hidden_channels1=64, hidden_channels2=32)
torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)


original_adj_matrix = fc_graph(4)
adj_matrix = original_adj_matrix.copy()
n_clients = 1
n_servers = 1

client_gateways, server_gateways = get_gw(adj_matrix, n_clients, n_servers)

env = None
######## CACHE ######
ip_to_node = None
node_to_ip = None

for epoch in range(20000):
    print('-'*20, "epoch : ", epoch, '-'*20)

    m = get_state(adj_matrix, client_gateways, server_gateways)

    actions, p, logits = agent.get_action(m, adj_matrix)
    adj_matrix = changeAdj(actions, original_adj_matrix)

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

    metrics, reward, e, q, (ip_to_node, node_to_ip) = env.step()
    log_prob = torch.log(p) * actions + torch.log(1-p) * (1-actions)
    loss = -torch.sum(log_prob * reward)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    print(
        f"Epoch {epoch}, Reward: {reward}, Loss: {loss.item()}, e: {e}, q: {q}")

    print("Sigmoid probabilities:", p.view(-1))
    print("Sampled actions:", actions.view(-1))

    if env is not None:
        ns.Simulator.Destroy()
        env = None
        client_gateways, server_gateways = get_gw(
            adj_matrix, n_clients, n_servers)
