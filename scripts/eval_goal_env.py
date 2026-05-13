import argparse
import time
from pathlib import Path
 
import numpy as np
from stable_baselines3 import SAC
 
from kernel_tuner_rl.goal_env import KernelTunerRLEnvGoal
from kernel_tuner_rl.cache_picker import choose_cache_file
from kernel_tuner_rl.backends import CacheBackend
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs="+", type=Path, default=None, help="Path(s) to cache file(s) from same kernel")
    parser.add_argument("--model", type=Path, default=None, help="Path to rl model")
    parser.add_argument("--desired-speedup", type=float, default=10.0,
                        help="Aspirational speedup target (unused when backend has precomputed stats)")
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
 
    model = args.model if args.model else f"models/sac_her_{paths[0].parent.name}"
 
    return paths, model, args.desired_speedup
 
 
N_EPISODES = 5
 
if __name__ == "__main__":
    cache_files, model_path, desired_speedup = parse_args()
    print(f"Using cache files: {cache_files}")
    # HER-trained models require an env at load time so the buffer can
    # reconstruct its goal/observation layout and compute_reward hook.
    load_env = KernelTunerRLEnvGoal(CacheBackend(str(cache_files[0])), desired_speedup)
    model = SAC.load(model_path, env=load_env)
 
    for cache_file in cache_files:
        backend = CacheBackend(str(cache_file))
        env = KernelTunerRLEnvGoal(backend, desired_speedup)
 
        all_times = sorted(
            t for t in (env.backend.get_exec_time(c) for c in env.backend.searchspace) if t != float("inf")
        )
        actual_best = all_times[0]
        median_exec_time = np.median(all_times)
        scale = median_exec_time - actual_best
        n_configs = len(all_times)
 
        start_time = time.time()
 
        per_episode_best = []
        per_episode_percentile = []
        overall_best_exec_time = float("inf")
        overall_best_config = None
 
        for ep in range(N_EPISODES):
            obs, info = env.reset(seed=ep)
            np.random.seed(ep)
            truncated = False
            terminated = False
            while not truncated and not terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
 
            ep_best = env.best_seen_exec_time
            per_episode_best.append(ep_best)
 
            n_better = sum(1 for t in all_times if t < ep_best)
            per_episode_percentile.append(n_better / n_configs * 100)
 
            if ep_best < overall_best_exec_time:
                overall_best_exec_time = ep_best
                overall_best_config = env.best_seen_config.copy()
 
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
