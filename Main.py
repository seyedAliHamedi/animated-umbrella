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

for epoch in range(10):

    subprocess.run(["python", "single.py", str(epoch)])
    middle.load_data()
    metric = middle.data['metrics']
    print('/'*80)
    print(metric)
    print('/'*80)
    actions, logits = agent.get_action(metric, adj_matrix)
    adj_matrix = changeAdj(actions, adj_matrix)
    middle.data['adj_matrix'] = adj_matrix
    middle.save_data()
    reward = middle.data['reward']
    loss = -torch.sum(torch.log10(logits) * reward)
    agent.optimizer.zero_grad()

    loss.backward()
    agent.optimizer.step()

    print("-"*20)
    print(f"LOSS : {loss}")
    print(f"Updated Adjacency Matrix: {adj_matrix}")

print("All epochs completed")
