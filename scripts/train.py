import os
import sys
import argparse
from pathlib import Path
import csv

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
from stable_baselines3.common.callbacks import BaseCallback

from kernel_tuner_rl.env import KernelTunerRLEnv
from kernel_tuner_rl.discrete_env import KernelTunerRLEnvDiscrete
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
    args = parser.parse_args()

    holdouts = []
    seed = 0

    if args.seed is not None:
        seed = args.seed

    if args.holdout:
        holdouts = args.holdout

    if args.train:
        return args.train, holdouts, seed

    selected = choose_cache_file()
    if selected is None:
        print("No cache file selected.")
        raise SystemExit(0)

    return [selected], holdouts, seed

def make_env(cache_file, index):
    log_dir = f"tmp/env_{index}/"
    os.makedirs(log_dir, exist_ok=True)
    def _init():
        backend = CacheBackend(str(cache_file))
        env = KernelTunerRLEnvDiscrete(backend)
        env = Monitor(
            env,
            log_dir,
            info_keywords=("best_seen_exec_time", "terminated_at_optimum"),
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class CallBack(BaseCallback):
    def __init__(self, eval_freq, train_files, holdout_files, save_path):
        super().__init__()
        self.eval_freq = eval_freq
        self.train_files = train_files
        self.holdout_files = holdout_files
        self.save_path = save_path
        
        with open(self.save_path, 'w', newline='') as save_file:
            writer = csv.writer(save_file)
            writer.writerow([
                "timestep", "gpu", "is_holdout", "episode", "gap", "best_exec_time",
                "entropy", "policy_loss", "value_loss", "clip_fraction", "mean_ep_return"
            ])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:

            log = self.model.logger.name_to_value
            entropy = log.get("train/entropy_loss", None)
            policy_loss = log.get("train/policy_gradient_loss", None)
            value_loss = log.get("train/value_loss", None)
            clip_fraction = log.get("train/clip_fraction", None)

            ep_info = self.model.ep_info_buffer
            mean_ep_return = np.mean([ep["r"] for ep in ep_info]) if ep_info else None

            with open(self.save_path, 'a', newline='') as save_file:
                writer = csv.writer(save_file)

                all_files = [(f, False) for f in self.train_files] + [(f, True) for f in self.holdout_files]
                N_EPISODES = 1
                for f, is_holdout in all_files:
                    backend = CacheBackend(str(f))
                    env = KernelTunerRLEnv(backend)
                    env = ActionMasker(env, mask_fn)

                    all_times = sorted(
                        t for t in (env.env.backend.get_exec_time(c) for c in env.env.backend.searchspace) if t != float("inf")
                    )
                    actual_best = all_times[0]
                    median_exec_time = np.median(all_times)
                    scale = median_exec_time - actual_best

                    for ep in range(N_EPISODES):
                        obs, info = env.reset(seed=ep)
                        np.random.seed(ep)
                        truncated = False
                        terminated = False
                        while not truncated and not terminated:
                            action, _ = self.model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                            obs, reward, terminated, truncated, info = env.step(action)

                        ep_best = env.env.best_seen_exec_time
                        gap = (ep_best - actual_best) / scale * 100

                        writer.writerow([
                            self.num_timesteps, f.name.split('.')[0], is_holdout, ep, gap, ep_best,
                            entropy, policy_loss, value_loss, clip_fraction, mean_ep_return
                        ])

        return True


if __name__ == "__main__":
    cache_files, holdout_files, seed = parse_args()

    kernel_name = None
    holdout_name = None
    try:
        kernel_name = get_kernel_name(cache_files)
        if holdout_files:
            holdout_name = get_kernel_name(holdout_files)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    model_save_path = f"models/ppo_{kernel_name}"
    os.makedirs('models/', exist_ok=True)

    env = DummyVecEnv([make_env(file, index) for index, file in enumerate(cache_files)])

    for index, file in enumerate(cache_files):
        e = env.envs[index].env.env
        print(f"\n--- {file.name.split('.')[0]} ---")
        if e.backend.has_precomputed_stats:
            print(f"Longest Execution Time: {e.max_exec_time}")
            print(f"Shortest Execution Time: {e.optimum_exec_time}")
        print(f"Number of Parameters: {e.number_of_parameters}")
        print(f"Number of Configurations in Search Space: {e.backend.get_searchspace_size()}\n")

    results_dir = f"results/{kernel_name}/seed_{seed}/"
    os.makedirs(results_dir, exist_ok=True)

    model = MaskablePPO(MaskableActorCriticPolicy, env, seed=seed, verbose=1, gamma=0.999, clip_range=0.2, ent_coef=0.01)
    myCallback = CallBack(25_000, cache_files, holdout_files, results_dir + "eval_results.csv")
    try:
        model.learn(total_timesteps=200_000)
    except Exception as e:
        print(f"learn() failed: {e}")
        raise

    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")


