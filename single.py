# single.py
import time
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


# epoch_number = int(sys.argv[1])
a = time.time()
run_single_epoch(epoch_number=1)
print(time.time()-a)
