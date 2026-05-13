import os
import sys
import argparse
from pathlib import Path
import csv
 
import numpy as np
 
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
 
from kernel_tuner_rl.goal_env import KernelTunerRLEnvGoal
from kernel_tuner_rl.cache_picker import choose_cache_file
from kernel_tuner_rl.backends import CacheBackend
from kernel_tuner_rl.utils import get_kernel_name
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=Path, default=None,
                        help="Path(s) to cache file(s) from same kernel")
    parser.add_argument("--holdout", nargs="+", type=Path, default=None,
                        help="Path(s) to holdout cache file(s) for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Seed for training")
    parser.add_argument("--desired-speedup", type=float, default=10.0,
                        help="Aspirational speedup target used at deployment time")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    args = parser.parse_args()
 
    holdouts = args.holdout or []
 
    if args.train:
        return args.train, holdouts, args.seed, args.desired_speedup, args.total_timesteps
 
    selected = choose_cache_file()
    if selected is None:
        print("No cache file selected.")
        raise SystemExit(0)
 
    return [selected], holdouts, args.seed, args.desired_speedup, args.total_timesteps
 
 
def make_env(cache_file, index, desired_speedup):
    log_dir = f"tmp/env_sac_{index}/"
    os.makedirs(log_dir, exist_ok=True)
 
    def _init():
        backend = CacheBackend(str(cache_file))
        env = KernelTunerRLEnvGoal(backend, desired_speedup)
        env = Monitor(
            env,
            log_dir,
            info_keywords=("best_seen_exec_time", "terminated_at_optimum"),
        )
        return env
 
    return _init
 
 
class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, train_files, holdout_files, save_path, desired_speedup):
        super().__init__()
        self.eval_freq = eval_freq
        self.train_files = train_files
        self.holdout_files = holdout_files
        self.save_path = save_path
        self.desired_speedup = desired_speedup
 
        with open(self.save_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestep", "gpu", "is_holdout", "episode", "gap", "best_exec_time",
                "actor_loss", "critic_loss", "ent_coef", "ent_coef_loss",
                "mean_ep_return",
            ])
 
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
 
        log = self.model.logger.name_to_value
        actor_loss = log.get("train/actor_loss")
        critic_loss = log.get("train/critic_loss")
        ent_coef = log.get("train/ent_coef")
        ent_coef_loss = log.get("train/ent_coef_loss")
 
        ep_info = self.model.ep_info_buffer
        mean_ep_return = np.mean([ep["r"] for ep in ep_info]) if ep_info else None
 
        with open(self.save_path, "a", newline="") as f:
            writer = csv.writer(f)
 
            all_files = (
                [(p, False) for p in self.train_files]
                + [(p, True) for p in self.holdout_files]
            )
            N_EPISODES = 1
            for path, is_holdout in all_files:
                backend = CacheBackend(str(path))
                env = KernelTunerRLEnvGoal(backend, self.desired_speedup)
 
                all_times = sorted(
                    t for t in (env.backend.get_exec_time(c) for c in env.backend.searchspace)
                    if t != float("inf")
                )
                actual_best = all_times[0]
                median_exec_time = np.median(all_times)
                scale = median_exec_time - actual_best
 
                for ep in range(N_EPISODES):
                    obs, info = env.reset(seed=ep)
                    terminated = False
                    truncated = False
                    while not (terminated or truncated):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
 
                    ep_best = env.best_seen_exec_time
                    gap = (ep_best - actual_best) / scale * 100
 
                    writer.writerow([
                        self.num_timesteps, path.name.split(".")[0], is_holdout, ep,
                        gap, ep_best,
                        actor_loss, critic_loss, ent_coef, ent_coef_loss,
                        mean_ep_return,
                    ])
 
        return True
 
 
if __name__ == "__main__":
    cache_files, holdout_files, seed, desired_speedup, total_timesteps = parse_args()
 
    try:
        kernel_name = get_kernel_name(cache_files)
        if holdout_files:
            _ = get_kernel_name(holdout_files)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
 
    os.makedirs("models/", exist_ok=True)
    model_save_path = f"models/sac_her_{kernel_name}"
 
    env = DummyVecEnv([make_env(f, i, desired_speedup) for i, f in enumerate(cache_files)])
 
    for i, f in enumerate(cache_files):
        e = env.envs[i].env
        print(f"\n--- {f.name.split('.')[0]} ---")
        if e.backend.has_precomputed_stats:
            print(f"Longest Execution Time:  {e.max_exec_time}")
            print(f"Shortest Execution Time: {e.optimum_exec_time}")
        print(f"Number of Parameters: {e.number_of_parameters}")
        print(f"Search Space Size:    {e.backend.get_searchspace_size()}\n")
 
    results_dir = f"results/{kernel_name}/sac_her_seed_{seed}/"
    os.makedirs(results_dir, exist_ok=True)

    longest_episode = int(max(e.env.max_steps for e in env.envs))
    learning_starts = env.num_envs * longest_episode + 100
 
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_starts=learning_starts,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.001,
        learning_rate=3e-4,
        seed=seed,
        verbose=1,
    )
 
    callback = EvalCallback(
        eval_freq=25_000,
        train_files=cache_files,
        holdout_files=holdout_files,
        save_path=results_dir + "eval_results.csv",
        desired_speedup=desired_speedup,
    )
 
    try:
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"learn() failed: {e}")
        raise
 
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
