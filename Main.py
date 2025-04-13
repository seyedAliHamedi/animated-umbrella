import os
import torch
import subprocess

from utils import *
import middle
from agent import Agent
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


n = 15
adj_matrix = middle.data['adj_matrix'] = fc_graph(n)
middle.save_data()
agent = Agent(num_node_features=24, hidden_channels1=64,
              hidden_channels2=32)

for epoch in range(20):

    subprocess.run(["python", "single.py", str(epoch)])
    middle.load_data()
    metric = middle.data['metrics']
    # print('/'*80)
    # print(metric)
    # print('/'*80)
# Prepare data
    data = agent.dict_to_data(adj_matrix, metric)

    # Forward pass
    logits = agent(data).view(-1)
    probs = torch.sigmoid(logits)

    # Sample actions
    dist = torch.distributions.Bernoulli(probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    # Update network state
    adj_matrix = changeAdj(actions, adj_matrix)
    middle.data['adj_matrix'] = adj_matrix
    middle.save_data()

    # Reward
    reward = middle.data['reward']
    if not torch.is_tensor(reward):
        reward = torch.tensor(reward, dtype=torch.float)

    # Loss (REINFORCE)
    loss = -(log_probs * reward).mean()

    # Backprop
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # Logging
    print("sigmoid", probs)
    print("agent actions", actions)
    print("Reward:", reward)
    print("LOSS:", loss.item())
    agent.optimizer.zero_grad()

    loss.backward()
    agent.optimizer.step()

    print("-"*20)
    print(f"LOSS : {loss}")
    print(f"Updated Adjacency Matrix: {adj_matrix}")

print("All epochs completed")
