# single.py
import sys
import json
from ns import ns
from rl_env import NetworkEnv
import os
import middle


def run_single_epoch(epoch_number):
    print(f"Running Epoch {epoch_number}")
    middle.load_data()
    adj_matrix = middle.data['adj_matrix']
    print("888888")
    print(adj_matrix)
    print("888888")
    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix
    )
    metrics, reward = env.step()
    middle.data['metrics'] = metrics
    middle.data['reward'] = reward
    middle.save_data()
    print(f"    Reward: {reward}")
    print(f"Epoch {epoch_number} completed")


# --- Entry point ---
epoch_number = int(sys.argv[1])

run_single_epoch(epoch_number)
