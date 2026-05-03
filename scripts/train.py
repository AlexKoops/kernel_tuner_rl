import os
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

from kernel_tuner_rl.env import KernelTunerRLEnv
from kernel_tuner_rl.cache_picker import choose_cache_file
from kernel_tuner_rl.backends import CacheBackend


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


def resolve_cache_file() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default=None, help="Path to cache file")
    args = parser.parse_args()

    if args.cache:
        return Path(args.cache)

    selected = choose_cache_file()
    if selected is None:
        print("No cache file selected.")
        raise SystemExit(0)

    return selected


if __name__ == "__main__":
    cache_file = resolve_cache_file()
    print(f"Using cache file: {cache_file}")

    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    backend = CacheBackend(str(cache_file))
    env = KernelTunerRLEnv(backend)
    env = Monitor(env, log_dir)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, gamma=0.999, clip_range=0.2, ent_coef=0.01)
    model.learn(total_timesteps=200_000)

    plot_results([log_dir], 200_000, results_plotter.X_EPISODES, "PPO kernel env")
    plt.show()

    save_dir = "models/"
    os.makedirs(save_dir, exist_ok=True)
    model.save(Path(save_dir) / "ppo_kernel_tuner")
