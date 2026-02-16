
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
import numpy as np

def test_evogym():
    print("Testing EvoGym installation...")
    
    # Create a random robot body
    body, connections = sample_robot((5,5))
    
    # Create the environment
    # Using render_mode=None to avoid GUI issues during automated testing
    try:
        env = gym.make('Walker-v0', body=body, render_mode=None)
        env.reset()
        print("Environment created successfully.")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return False

    # Run for a few steps
    try:
        for _ in range(100):
            action = env.action_space.sample()
            ob, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        print("Simulation ran successfully for 100 steps.")
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False
        
    env.close()
    return True

if __name__ == '__main__':
    if test_evogym():
        print("\nEvoGym is correctly installed and working!")
    else:
        print("\nEvoGym installation validation failed.")
