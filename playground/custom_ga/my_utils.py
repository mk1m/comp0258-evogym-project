import math
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform
import numpy as np

class Structure():
    def __init__(self, body, connections, label):
        self.body = body
        self.connections = connections
        
        # CHANGED: Use a dictionary for rewards
        self.rewards = {} 
        self.fitness = 0
        
        self.is_survivor = False
        self.prev_gen_label = 0
        self.label = label

    def compute_fitness(self):
        # CHANGED: Weighted Sum
        # Define weights here (or pass them in). 
        # 1.0 = Maximize, -1.0 = Minimize, 0.0 = Ignore
        # Example: maximize Walker, minimize Carrier
        TASK_WEIGHTS = {
            'Walker-v0': 1.0,
            'Carrier-v0': -1.0, # Negative weight to minimize this task
            'Climber-v0': 1.0
        }
        
        self.fitness = 0
        if not self.rewards:
            pass # Keep 0
        else:
            for task, reward in self.rewards.items():
                weight = TASK_WEIGHTS.get(task, 1.0) # Default to 1.0 if not specified
                self.fitness += weight * reward
                
        return self.fitness

    def set_reward(self, task_name, reward):
        # CHANGED: Store reward by task name
        self.rewards[task_name] = reward
        self.compute_fitness()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.rewards}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()

# --- REST IS COPIED AS-IS FROM evogym/examples/utils/algo_utils.py ---

class TerminationCondition():
    def __init__(self, max_iters):
        self.max_iters = max_iters
    def __call__(self, iters):
        return iters >= self.max_iters
    def change_target(self, max_iters):
        self.max_iters = max_iters

def mutate(child, mutation_rate=0.1, num_attempts=10):
    pd = get_uniform(5)  
    pd[0] = 0.6 
    for n in range(num_attempts):
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: 
                    child[i][j] = draw(pd)
        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))
    return None

def get_percent_survival_evals(curr_eval, max_evals):
    low = 0.0
    high = 0.6
    return ((max_evals-curr_eval-1)/(max_evals-1)) * (high-low) + low

def pretty_print(list_org, max_name_length=30):
    list_formatted = []
    for i in range(len(list_org)//4 +1):
        list_formatted.append([])
    for i in range(len(list_org)):
        row = i%(len(list_org)//4 +1)
        list_formatted[row].append(list_org[i])
    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " "*(max_name_length - len(str(el)))
        print(out)
