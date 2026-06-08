"""
Training script for PPO-based kernel tuning agents.

Trains a MaskablePPO agent on one or more GPU kernel cache files simultaneously.
The trained model is saved to models/<kernel_name>/<agent>_<tag>_ppo_seed<seed>.

Usage examples:
    python scripts/train.py --agent hamming --train clean_cache/<kernel_name>/*.json --seed 0 --total-timesteps 600000 --tag example
    python scripts/train.py --agent perparameter --train clean_cache/<kernel_name>/*.json --seed 0 --total-timesteps 2000000 --tag example
"""
import os
import sys
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from kernel_tuner_rl.envs import KernelTunerRLEnvHamming, KernelTunerRLEnvPerparameter
from kernel_tuner_rl.backends import CacheBackend
from kernel_tuner_rl.utils import get_kernel_name


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Extracts the action mask from the unwrapped environment to use with MaskablePPO

    Parameters:
        - env: the gymnasium environment from which the action_masks() function will be used

    Return:
        the np.ndarray filled with booleans, created by the action_masks() from env
    """
    return env.unwrapped.action_masks()

def parse_args():
    """
        ArgumentParser used to create the training procedure.
        --agent is a string of the agent that will be trained
        --train is the path or paths to the cache files used to train the agent on 
        --seed is the desired seed to set during training
        --total-timesteps is the timesetp budget for the whole training
        --tag can be used to give the agent file an extra label to separete it from other models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="hamming", help="Agent name string, choose between 'hamming' or 'perparameter' ")
    parser.add_argument("--train", nargs="+", type=Path, default=None, help="Path(s) to cache file(s) from same kernel")
    parser.add_argument("--seed", type=int, default=None, help="Seed for training")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional label for this run (e.g. 'gpufeat'). "
                             "Folded into the model and results paths.")

    args = parser.parse_args()

    train = []
    seed = 0

    if args.seed is not None:
        seed = args.seed

    if args.train:
        train = args.train

    agent = args.agent

    if agent not in ("hamming", "perparameter"):
        print(f"Error: --agent must be 'hamming' or 'perparameter', got '{agent}'")
        sys.exit(1)

    return agent, train, seed, args.total_timesteps, args.tag

def make_env(agent, cache_file, index):
    """
        Factory function to create a new environment for each cache_file used for training.
        Needed because DummyVecEnv expects a list of callables, not environments

        Monitor logs episode statistics, ActionMasker enables action masking for MaskablePPO
    """
    log_dir = f"tmp/env_{index}/"
    os.makedirs(log_dir, exist_ok=True)
    def _init():
        RLEnv = KernelTunerRLEnvPerparameter if agent == 'perparameter' else KernelTunerRLEnvHamming
        backend = CacheBackend(str(cache_file))
        env = RLEnv(backend)
        env = Monitor(env, log_dir,
                      info_keywords=("best_seen_exec_time", "terminated_at_optimum"))
        env = ActionMasker(env, mask_fn)
        return env
    return _init


if __name__ == "__main__":
    agent, cache_files, seed, total_timesteps, tag = parse_args()

    kernel_name = None
    try:
        kernel_name = get_kernel_name(cache_files)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Link the model tag to the agent name
    if tag:
        model_save_path = f"models/{kernel_name}/{agent}_{tag}_ppo_seed{seed}"
        results_dir = f"results/{kernel_name}/{agent}_{tag}/seed_{seed}/"
    else:
        model_save_path = f"models/{kernel_name}/{agent}_ppo_seed{seed}"
        results_dir = f"results/{kernel_name}/{agent}/seed_{seed}/"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create the vector environment to handle multiple file inputs
    env = DummyVecEnv([make_env(agent, file, i) for i, file in enumerate(cache_files)])

    # Print info about searchspaces being trained on
    for index, file in enumerate(cache_files):
        # unwrap Monitor and ActionMasker wrappers to access the base environment
        e = env.envs[index].env.env
        print(f"\nFile: {file.name.split('.')[0]}")
        if e.backend.has_precomputed_stats:
            print(f"Longest Execution Time: {e.max_exec_time}")
            print(f"Shortest Execution Time: {e.optimum_exec_time}")
        print(f"Number of Parameters: {e.number_of_parameters}")
        print(f"Number of Configurations in Search Space: {e.backend.get_searchspace_size()}\n")

    # gamma=0.999: high discount to keep terminal reward significant over 1000 step episodes
    # ent_coef=0.01: encourages exploration to avoid getting stuck in local minima
    # set verbose=1 to print training information during training
    model = MaskablePPO(MaskableActorCriticPolicy, env, seed=seed, verbose=0, gamma=0.999, clip_range=0.2, ent_coef=0.01)

    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except Exception as e:
        print(f"learn() failed: {e}")
        raise

    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
