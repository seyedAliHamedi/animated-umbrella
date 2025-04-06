from ns import ns
from rl_env import NetworkEnv
import traceback

def main():
    env = NetworkEnv(simulation_duration=1000)  
    action = {} 
    next_state, reward, done, info = env.step(action)
    print(f"Step successful, reward: {reward}")

    print("Test completed successfully!")
        
        
if __name__ == "__main__":
    main()