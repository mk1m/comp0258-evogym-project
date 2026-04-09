import os
import numpy as np
import shutil
import random
import math
import argparse
from typing import List

from ppo.run import run_ppo
import evogym.envs
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, Structure

import warnings
warnings.filterwarnings("ignore")

def run_different_ppo(
    args: argparse.Namespace,
):
    
    print()
    
    exp_name, env_name, pop_size, num_cores, structures_path = (
        args.exp_name,
        args.env_name,
        args.pop_size,
        args.num_cores,
        args.structures_path,
    )

    ### MANAGE DIRECTORIES ###
    home_path = os.path.join("saved_data", exp_name)
    print(f"Starting {exp_name}")

    ### DEFINE TERMINATION CONDITION ###

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        else:
            return


    ### GENERATE // GET INITIAL POPULATION ###
    structures: List[Structure] = []
    population_structure_hashes = {}
    generation = 0

    #read status from file

    for i in range(pop_size):
        save_path_structure = os.path.join(structures_path, str(i) + ".npz")
        np_data = np.load(save_path_structure)
        structure_data = []
        for key, value in np_data.items():
            structure_data.append(value)
        structure_data = tuple(structure_data)
        population_structure_hashes[hashable(structure_data[0])] = True
        structures.append(Structure(*structure_data, i))



        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join("saved_data", exp_name,"generation_" + str(generation), "structure")
        save_path_controller = os.path.join("saved_data", exp_name,"generation_" + str(generation), "controller")
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

        ### SAVE POPULATION DATA ###
    for i in range(len(structures)):
        temp_path = os.path.join(save_path_structure, str(structures[i].label))
        np.savez(temp_path, structures[i].body, structures[i].connections)

    ### TRAIN GENERATION

    #better parallel
    group = mp.Group()
    for structure in structures:
            
        ppo_args = (args,structure.body, env_name, save_path_controller, f'{structure.label}', structure.connections)
        group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
                

    group.run_jobs(num_cores)

    #not parallel
    #for structure in structures:
    #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

    ### COMPUTE FITNESS, SORT, AND SAVE ###
    for structure in structures:
        structure.compute_fitness()

    structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

    #SAVE RANKING TO FILE
    temp_path = os.path.join("saved_data", exp_name,"generation_" + str(generation), "output.txt")
    f = open(temp_path, "w")

    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
    f.write(out)
    f.close()

    print('DONE')
