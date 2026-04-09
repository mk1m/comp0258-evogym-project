import random
import numpy as np
import argparse

from ga.different_ppo import run_different_ppo
from ppo.args import add_ppo_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    parser = argparse.ArgumentParser(description='Arguments for ppo script')
    parser.add_argument('--exp-name', type=str, default='test_ppo', help='Name of the experiment (default: test_ppo)')
    parser.add_argument('--env-name', type=str, default='Walker-v0', help='Name of environment (default: Walker-v0)')
    parser.add_argument('--pop-size', type=int, default=15, help='Population size (default: 15)')
    parser.add_argument('--num-cores', type=int, default=8, help='Number of robots to evaluate simultaneously (default: 8)')
    parser.add_argument('--structures-path', type=str, default='saved_data/test_ga/structures', help='Save path for robot structures (default: saved_data/test_ga/structures)')
    add_ppo_args(parser)
    args = parser.parse_args()
    
    run_different_ppo(args)
