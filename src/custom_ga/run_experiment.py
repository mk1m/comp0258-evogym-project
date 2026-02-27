import sys
import os
import argparse
import random
import numpy as np

# Add project root to path so we can import evogym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Add evogym/examples to path so we can import ppo, utils, etc if needed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'evogym', 'examples')))

from my_ga import run_ga
from ppo.args import add_ppo_args

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    parser = argparse.ArgumentParser(description='Run Custom Multi-Task GA')
    parser.add_argument('--exp-name', type=str, default='test_custom_ga', help='Name of the experiment')
    parser.add_argument('--pop-size', type=int, default=4, help='Population size')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5), help='Shape of the structure')
    parser.add_argument('--max-evaluations', type=int, default=8, help='Maximum number of robots evaluation')
    parser.add_argument('--num-cores', type=int, default=2, help='Number of robots to evaluate simultaneously')
    
    # NEW ARGUMENT: List of environments
    parser.add_argument('--env-names', nargs='+', default=['Walker-v0', 'Carrier-v0'], 
                        help='List of environments to train on (e.g. Walker-v0 Carrier-v0)')
    
    add_ppo_args(parser)
    args = parser.parse_args()
    
    # Run the custom GA
    run_ga(args)
