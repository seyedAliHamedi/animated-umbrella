# main.py
import subprocess

for epoch in range(1, 6):
    subprocess.run(["python", "single.py", str(epoch)])
    print(f"Completed Epoch {epoch}")

print("All epochs completed")