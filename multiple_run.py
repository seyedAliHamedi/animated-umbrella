import os
import subprocess
import time

n = 100
if os.path.exists("./agent_weights.pth"):
    os.remove("./agent_weights.pth")

for i in range(n):
    print(f"\n=== Run {i + 1}/{n} ===")
    subprocess.run(["python", "Main.py"])
