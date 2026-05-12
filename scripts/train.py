import os
import sys
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from kernel_tuner_rl.env import KernelTunerRLEnv
from kernel_tuner_rl.cache_picker import choose_cache_file
from kernel_tuner_rl.backends import CacheBackend
from kernel_tuner_rl.utils import get_kernel_name


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


def parse_args() -> list[Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=Path, default=None, help="Path(s) to cache file(s) from same kernel")
    args = parser.parse_args()

    if args.train:
        return args.train

    selected = choose_cache_file()
    if selected is None:
        print("No cache file selected.")
        raise SystemExit(0)

    return [selected]

def make_env(cache_file, index):
    log_dir = f"tmp/env_{index}/"
    os.makedirs(log_dir, exist_ok=True)
    def _init():
        backend = CacheBackend(str(cache_file))
        env = KernelTunerRLEnv(backend)
        env = Monitor(
            env,
            log_dir,
            info_keywords=("best_seen_exec_time", "terminated_at_optimum"),
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init



if __name__ == "__main__":
    cache_files = parse_args()

    kernel_name = None
    try:
        kernel_name = get_kernel_name(cache_files)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    model_save_path = f"models/ppo_{kernel_name}"
    os.makedirs('models/', exist_ok=True)
    print(f"Using cache files: {cache_files}")

    env = DummyVecEnv([make_env(file, index) for index, file in enumerate(cache_files)])

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, gamma=0.999, clip_range=0.2, ent_coef=0.01)
    model.learn(total_timesteps=100_000)


    log_dirs = [f"tmp/env_{index}/" for index in range(len(cache_files))]
    plot_results(log_dirs, 100_000, results_plotter.X_TIMESTEPS, f"PPO {kernel_name}")
    plt.show()

    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")


