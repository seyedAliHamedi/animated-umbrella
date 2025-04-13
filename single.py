# single.py
import sys
import json
from ns import ns
from rl_env import NetworkEnv
import os

os.environ["CPPYY_UNCAUGHT_QUIET"] = "1"


def get_adj():
    adj_matrix = []
    n = 16
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        adj_matrix.append(row)
    return adj_matrix


def run_single_epoch(epoch_number, adj_matrix, output_path):
    print(f"Running Epoch {epoch_number}")
    adj_matrix = middle.adj_matrix
    env = NetworkEnv(
        simulation_duration=50,
        adj_matrix=adj_matrix
    )
    metrics, reward = env.step()
    middle.metrics = metrics
    middle.reward = reward
    print(f"    Reward: {reward}")
    print(f"Epoch {epoch_number} completed")

    # Overwrite the file with updated matrix
    with open(output_path, "w") as f:
        json.dump(adj_matrix, f)


# --- Entry point ---
epoch_number = int(sys.argv[1])
matrix_path = sys.argv[2]

# Load the matrix from the shared JSON file
with open(matrix_path, "r") as f:
    adj_matrix = json.load(f)

run_single_epoch(epoch_number, adj_matrix, matrix_path)
