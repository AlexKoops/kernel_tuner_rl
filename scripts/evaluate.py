import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from kernel_tuner_rl.discrete_env import KernelTunerRLEnvDiscrete
from kernel_tuner_rl.cache_picker import choose_cache_file
from kernel_tuner_rl.backends import CacheBackend


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs="+", type=Path, default=None, help="Path(s) to cache file(s) from same kernel")
    parser.add_argument("--model", type=Path, default=None, help="Path to rl model")
    args = parser.parse_args()

    paths = []

    if args.test:
        paths = args.test
    else:
        path = choose_cache_file()
        if path is None:
            print("No cache file selected.")
            raise SystemExit(0)
        paths.append(path)

    model = args.model if args.model else f"models/ppo_{paths[0].parent.name}"

    return paths, model


N_EPISODES = 5

if __name__ == "__main__":
    cache_files, model_path = parse_args()
    print(f"Using cache files: {cache_files}")
    model = MaskablePPO.load(model_path)

    for cache_file in cache_files:
        backend = CacheBackend(str(cache_file))
        env = KernelTunerRLEnvDiscrete(backend)

        env = ActionMasker(env, mask_fn)

        all_times = sorted(
            t for t in (env.env.backend.get_exec_time(c) for c in env.env.backend.searchspace) if t != float("inf")
        )
        actual_best = all_times[0]
        median_exec_time = np.median(all_times)
        scale = median_exec_time - actual_best
        n_configs = len(all_times)

        start_time = time.time()  # start timing from RL part

        per_episode_best = []
        per_episode_percentile = []
        overall_best_exec_time = float("inf")
        overall_best_config = None

        for ep in range(N_EPISODES):
            ep_seed = int(time.time() * 1000 + ep)

            obs, info = env.reset(seed=ep_seed)
            truncated = False
            terminated = False
            while not truncated and not terminated:
                action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

            ep_best = env.env.best_seen_exec_time
            per_episode_best.append(ep_best)

            n_better = sum(1 for t in all_times if t < ep_best)
            per_episode_percentile.append(n_better / n_configs * 100)

            if ep_best < overall_best_exec_time:
                overall_best_exec_time = ep_best
                overall_best_config = env.env.best_seen_config.copy()

        end_time = time.time() - start_time

        per_episode_best = np.array(per_episode_best)
        per_episode_percentile = np.array(per_episode_percentile)

        gaps = (per_episode_best - actual_best) / scale * 100
        overall_gap = (overall_best_exec_time - actual_best) / scale * 100

        overall_n_better = sum(1 for t in all_times if t < overall_best_exec_time)
        overall_percentile = overall_n_better / n_configs * 100

        print()
        print("------------------------------------")
        print("RESULT STATISTICS:")
        print("------------------------------------")
        print()
        print(f"Cache optimum:             {actual_best:.4f}ms")
        print(f"Search space median:       {median_exec_time:.4f}ms")
        print(f"Search space size:         {n_configs} configs")
        print()
        print(
            f"Best across {N_EPISODES} episodes:   {overall_best_exec_time:.4f}ms   (gap optimum-median {overall_gap:.2f}%,  top {100 - overall_percentile:.2f}%  —  {overall_n_better} configs better)"
        )
        print(
            f"Mean per-episode best:     {per_episode_best.mean():.4f}ms   (gap optimum-median {gaps.mean():.2f}%,  top {100 - per_episode_percentile.mean():.2f}%)"
        )
        print(
            f"Median per-episode best:   {np.median(per_episode_best):.4f}ms   (gap optimum-median {np.median(gaps):.2f}%,  top {100 - np.median(per_episode_percentile):.2f}%)"
        )
        print(f"Std across episodes:       {per_episode_best.std():.4f}ms")
        print()
        print(f"Best config found: {overall_best_config}")
        print()
        print("--- %s seconds ---" % end_time)
