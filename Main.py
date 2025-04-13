# main.py
import subprocess
import json
import os

adj_matrix = []
n = 16
for i in range(n):
    row = []
    for j in range(n):
        row.append(0 if i == j else 1)
    adj_matrix.append(row)


def changeAdj(action):
    return action


output_file = "adj_matrix.json"
agent = 0
for epoch in range(1, 2):
    # Save current adj_matrix to file before running
    with open(output_file, "w") as f:
        json.dump(adj_matrix, f)
    metric = middle.metric
    action = agent(metric)
    adj_matrix = changeAdj(action)
    subprocess.run(["python", "single.py", str(
        epoch), output_file])  # pass adj
    reward = middle.reward
    loss = 0
    loss.backprop()

    # Read back the updated matrix (overwrite behavior)
    with open(output_file, "r") as f:
        adj_matrix = json.load(f)
    print("-"*20)
    print(f"Updated Adjacency Matrix: {adj_matrix}")
    # print(f"Completed Epoch {epoch}, updated matrix loaded")

print("All epochs completed")
