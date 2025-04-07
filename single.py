# single_epoch.py
from ns import ns
from rl_env import NetworkEnv

def run_single_epoch(epoch_number):
    print(f"Running Epoch {epoch_number}")
    env = NetworkEnv(
        topology_base_network="192.166.1.0/24",
        simulation_duration=2500,
        max_steps=100,
        each_step_duration=20
    )
    
    for step in range(100):
        print(f"  Step {step+1}/100")
        action = {}
        _, reward, done, _ = env.step(action)
        print(f"    Reward: {reward}")
        
        if done:
            break
    
    print(f"Epoch {epoch_number} completed")
    
if __name__ == "__main__":
    import sys
    epoch_number = int(sys.argv[1])
    run_single_epoch(epoch_number)