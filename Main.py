import os
import torch
from ns import ns
from rl_env import NetworkEnv
from utils import *
from agent import Agent
os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


n = 4

# original_adj_matrix = fc_graph(n)
original_adj_matrix = [[0,0,0,1],
                       [0,0,0,1],
                       [0,0,0,1],
                       [1,1,1,0]]
adj_matrix = original_adj_matrix.copy()


agent = Agent(num_node_features=24, hidden_channels1=64, hidden_channels2=32)
env = None
prev_actions = None

for epoch in range(2000):
    if env is not None:
        ns.Simulator.Destroy()
        env = None

    # === Create new environment using current adj_matrix ===
    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix,
        orinigal_adj_matrix=original_adj_matrix,
        n_clients=1,
        n_servers=1
    )

    # === Step 1: Run the simulation with current graph ===
    metrics, reward = env.step()

    # === Step 2: If we have previous actions, use them to train ===
    if prev_actions is not None:
        log_prob = prev_actions * torch.log(prev_p + 1e-10) + \
                   (1 - prev_actions) * torch.log(1 - prev_p + 1e-10)
        loss = -torch.sum(log_prob) * reward

        # Optionally skip training if reward is corrupted
        if reward < -0.01 and torch.all(prev_actions == 0):
            print("Skipping training due to invalid reward on fully active graph")
        else:
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            print(f"Epoch {epoch}, Reward: {reward}, Loss: {loss.item()}")
    else:
        print(f"Epoch {epoch}, First epoch – no previous actions.")

    # === Step 3: Generate current actions and update adj_matrix ===
    actions, p, logits = agent.get_action(metrics, adj_matrix)
    print("Sigmoid probabilities:", p)
    print("Sampled actions:", actions)

    adj_matrix = changeAdj(actions, original_adj_matrix)

    # === Step 4: Store current actions for next epoch ===
    prev_actions = actions
    prev_p = p