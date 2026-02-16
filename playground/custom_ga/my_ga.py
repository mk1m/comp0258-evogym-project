import os
import numpy as np
import shutil
import random
import math
import argparse

# CHANGED: Import from our local my_utils
from ppo.run import run_ppo
import evogym.envs
from evogym import sample_robot, hashable
import utils.mp_group as mp
from my_utils import get_percent_survival_evals, mutate, Structure

def run_ga(args):
    print()
    
    exp_name = args.exp_name
    pop_size = args.pop_size
    structure_shape = args.structure_shape
    max_evaluations = args.max_evaluations
    num_cores = args.num_cores
    
    # CHANGED: Allow multiple environments (tasks)
    # We expect args.env_names to be a list, e.g., ['Walker-v0', 'Carrier-v0']
    # If standard args are passed (string), convert to list
    if hasattr(args, 'env_names') and args.env_names:
        env_names = args.env_names
    else:
        env_names = [args.env_name]

    print(f"Running GA on tasks: {env_names}")

    home_path = os.path.join("saved_data", exp_name)
    start_gen = 0
    is_continuing = False

    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    # Metadata storage (simplified)
    if not is_continuing:
        try: os.makedirs(os.path.join("saved_data", exp_name))
        except: pass

    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0
    
    # Generate initial population
    if not is_continuing: 
        for i in range (pop_size):
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)
            structures.append(Structure(*temp_structure, i))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1
    else:
        # (Handling continuation omitted for brevity in this custom script, 
        # but could be added back if needed)
        pass

    while True:
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))

        save_path_structure = os.path.join("saved_data", exp_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join("saved_data", exp_name, "generation_" + str(generation), "controller")
        
        try: os.makedirs(save_path_structure)
        except: pass
        try: os.makedirs(save_path_controller)
        except: pass

        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        # CHANGED: Training loop for MULTIPLE tasks
        group = mp.Group()
        for structure in structures:
            if structure.is_survivor:
                print(f'Skipping training for survivor {structure.label}')
                # Copy controller logic would go here, but omitted for simplicity in this demo
            else:
                # Launch a PPO job for EVERY task
                for env_name in env_names:
                    # Unique save path for each task's controller
                    controller_path = save_path_controller
                    controller_label = f'{structure.label}_{env_name}'
                    
                    ppo_args = (args, structure.body, env_name, controller_path, controller_label, structure.connections)
                    
                    # Callback needs to know WHICH task this reward is for
                    # We use a default arg in lambda to capture the value of env_name
                    callback_fn = lambda r, t=env_name, s=structure: s.set_reward(t, r)
                    
                    group.add_job(run_ppo, ppo_args, callback=callback_fn)

        group.run_jobs(num_cores)

        # Compute Fitness
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        # Output to file
        temp_path = os.path.join("saved_data", exp_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")
        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\t\t" + str(structure.rewards) + "\n"
        f.write(out)
        f.close()

        if num_evaluations >= max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])

        survivors = structures[:num_survivors]
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:
            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)
            if child != None and hashable(child[0]) not in population_structure_hashes:
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]
        generation += 1
