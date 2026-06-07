
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

from kernel_tuner_rl.big_env import KernelTunerRLEnvBig
from kernel_tuner_rl.cache_picker import choose_cache_file
from kernel_tuner_rl.backends import CacheBackend
from kernel_tuner_rl.utils import get_kernel_name


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=Path, default=None, help="Path(s) to cache file(s) from same kernel")
    parser.add_argument("--holdout", nargs="+", type=Path, default=None, help="Path(s) to holdout cache file(s) for evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Seed for training")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional label for this run (e.g. 'gpufeat'). "
                             "Folded into the model and results paths.")

    args = parser.parse_args()

    holdouts = []
    seed = 0

    if args.seed is not None:
        seed = args.seed

    if args.holdout:
        holdouts = args.holdout

    if args.train:
        return args.train, holdouts, seed, args.total_timesteps, args.tag

    selected = choose_cache_file()
    if selected is None:
        print("No cache file selected.")
        raise SystemExit(0)

    return [selected], holdouts, seed, args.total_timesteps, args.tag

def make_env(cache_file, index):
    log_dir = f"tmp/env_{index}/"
    os.makedirs(log_dir, exist_ok=True)
    def _init():
        backend = CacheBackend(str(cache_file))
        env = KernelTunerRLEnvBig(backend)
        env = Monitor(env, log_dir,
                      info_keywords=("best_seen_exec_time", "terminated_at_optimum"))
        env = ActionMasker(env, mask_fn)
        return env
    return _init


if __name__ == "__main__":
    cache_files, holdout_files, seed, total_timesteps, tag = parse_args()

    kernel_name = None
    holdout_name = None
    holdout_files = []
    try:
        kernel_name = get_kernel_name(cache_files)
        if holdout_files:
            holdout_name = get_kernel_name(holdout_files)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if tag:
        model_save_path = f"models/{kernel_name}/{tag}_ppo_seed{seed}"
        results_dir = f"results/{kernel_name}/{tag}/seed_{seed}/"
    else:
        model_save_path = f"models/{kernel_name}/ppo_seed{seed}"
        results_dir = f"results/{kernel_name}/seed_{seed}/"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    env = DummyVecEnv([make_env(file, i) for i, file in enumerate(cache_files)])

    for index, file in enumerate(cache_files):
        e = env.envs[index].env.env
        print(f"\n File: {file.name.split('.')[0]}")
        if e.backend.has_precomputed_stats:
            print(f"Longest Execution Time: {e.max_exec_time}")
            print(f"Shortest Execution Time: {e.optimum_exec_time}")
        print(f"Number of Parameters: {e.number_of_parameters}")
        print(f"Number of Configurations in Search Space: {e.backend.get_searchspace_size()}\n")

    model = MaskablePPO(MaskableActorCriticPolicy, env, seed=seed, verbose=0, gamma=0.999, clip_range=0.2, ent_coef=0.01)

    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except Exception as e:
        print(f"learn() failed: {e}")
        raise

    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
