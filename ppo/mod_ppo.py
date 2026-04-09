import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import evogym.envs
from ppo.eval import eval_policy
from ppo.callback import EvalCallback

import warnings
warnings.filterwarnings("ignore")

def run_ppo(
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """
    Run ppo and return the best reward achieved during evaluation.
    """
    
    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    })
    
    # Eval Callback
    callback = EvalCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_every=1e4,
        n_evals=4,
        n_envs=4,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        verbose=0,
    )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=4,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        clip_range=0.1
    )
    model.learn(
        total_timesteps=6e5,
        callback=callback,
        log_interval=50
    )
    
    return callback.best_reward