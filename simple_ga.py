import gymnasium as gym
import evogym.envs
from evogym import sample_robot
import numpy as np

def run_simple_ga():
    print("--- Starting Simple Random Search (GA Placeholder) ---")

    # 1. Undefined parameters
    pop_size = 5
    structure_shape = (5, 5)
    max_steps = 500

    # 2. Generate initial population of random robots
    # A robot is defined by its (body, connections)
    population = []
    for i in range(pop_size):
        body, connections = sample_robot(structure_shape)
        population.append({'body': body, 'connections': connections, 'id': i})

    # 3. Evaluate each robot
    print(f"\nEvaluating population of {pop_size} robots...")
    
    # Create the environment once
    # We use 'Walker-v0', a standard task
    env = gym.make('Walker-v0', body=population[0]['body'], render_mode=None) 

    for robot in population:
        # Reset environment with the specific robot body
        env = gym.make('Walker-v0', body=robot['body'], render_mode=None)
        obs, _ = env.reset()
        
        total_reward = 0
        for _ in range(max_steps):
            # In a real GA, you'd use a policy here. 
            # For this simple demo, we just sample random actions.
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        robot['fitness'] = total_reward
        env.close()
        print(f"Robot {robot['id']}: Fitness = {total_reward:.2f}")

    # 4. specific robot selection (Simple Elitism)
    best_robot = max(population, key=lambda x: x['fitness'])
    print(f"\nBest Robot ID: {best_robot['id']} with Fitness: {best_robot['fitness']:.2f}")

    # 5. Visualize the best robot
    print("\nVisualizing the best robot...")
    try:
        env = gym.make('Walker-v0', body=best_robot['body'], render_mode='human')
        obs, _ = env.reset()
        while True:
            action = env.action_space.sample() # Replace with trained policy if available
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
    except KeyboardInterrupt:
        print("Visualization stopped.")
    finally:
        env.close()

if __name__ == '__main__':
    run_simple_ga()
