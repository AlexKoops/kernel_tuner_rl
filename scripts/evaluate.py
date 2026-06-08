"""
Evaluation script for PPO-based kernel tuning agents.

Evaluates MaskablePPO agents on one or more cache files

Usage example:
    python scripts/evaluate.py --test clean_cache/convolution_milo/*.json --models models/convolution_milo/ppo_seed0 --agent hamming --episodes 10
"""
import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from kernel_tuner_rl.envs import KernelTunerRLEnvSmall, KernelTunerRLEnvBig
from kernel_tuner_rl.backends import CacheBackend
from kernel_tuner_rl.utils import random_baseline

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
    ArgumentParser used to create the evaluation procedure.
    --test is the path or paths to the cache files used to test the agent on 
    --model are the model files or a directory in which to find the models
    --agent is a string of the agent that will be evaluated
    --tag can be used to give the agent file an extra label to separete it from other models
    --episodes is the amount of episodes to run per file for evaluation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs="+", type=Path, default=None,
                        help="Path(s) to cache file(s) from same kernel")
    parser.add_argument("--models", nargs="+", type=Path, default=None,
                        help="One or more model files, or a directory of models. "
                             "If omitted, all seed models in the kernel's model dir are used.")
    parser.add_argument("--agent", type=str, default=None, help="Agent type to use during evaluation (hamming or perparameter) will default to hamming")
    parser.add_argument("--tag", type=str, default=None,
                        help="If set, only models whose name starts with this tag are used "
                             "when auto-discovering (e.g. 'hamming_example').")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per file")
    args = parser.parse_args()

    paths = args.test 
    
    return paths, args.models, args.agent, args.tag, args.episodes


def resolve_models(models_arg, agent, tag, cache_files):
    """
    Helper function to resolve the models based on the agent and tag arguments
    If no model argument is given, models will be selected based on kernel name from cache files

    Parameters:
        - models_arg: the model files or directory containing the models to evaluate
        - agent: used agent name string for the model
        - tag: tag added to the agent during training
        - cache_files: paths to cache_files which will be used to evaluate the models on

    Return:
        list of model paths which will be evaluated
    """

    if tag and agent:
        glob_pat = f"{agent}_{tag}_ppo_seed*"
    elif tag:
        glob_pat = f"{tag}_ppo_seed*"
    elif agent: 
        glob_pat = f"{agent}_ppo_seed*"
    else:
        glob_pat = "*ppo_seed*"

    if not models_arg:
        kernel = cache_files[0].parent.name
        model_dir = Path(f"models/{kernel}")
        return sorted(model_dir.glob(glob_pat))

    resolved = []
    for model in models_arg:
        if model.is_dir():
            resolved.extend(sorted(model.glob(glob_pat)))
        else:
            resolved.append(model)
    return resolved


def evaluate_one(model, cache_file, n_episodes, agent):
    """
    Evaluates a single model for n episodes on one cache file

    Parameters:
        - model: the model to evaluate
        - cache_file: the cachefile the model will be evaluated on 
        - n_episodes: the amount of episodes used for evaluation

    Return:
        dict with evaluation metrics used for performance display
    """
    RLEnv = KernelTunerRLEnvBig if agent == 'perparameter' else KernelTunerRLEnvSmall
    backend = CacheBackend(str(cache_file))
    env = RLEnv(backend)
    env = ActionMasker(env, mask_fn)

    all_times = sorted(
        t for t in (env.env.backend.get_exec_time(c) for c in env.env.backend.searchspace)
        if t != float("inf")
    )
    actual_best = all_times[0]
    median_exec_time = float(np.median(all_times))
    scale = median_exec_time - actual_best
    n_configs = len(all_times)

    # calculate a baseline to compare agent against
    baseline = random_baseline(
        backend,
        n_samples=env.env.max_steps,
        n_trials=20,
        seed=0,
        actual_best=actual_best,
        scale=scale,
    )

    per_episode_best = []
    overall_best_exec_time = float("inf")
    overall_best_config = None

    # run N episodes with agent
    for ep in range(n_episodes):
        ep_seed = int(time.time() * 1000 + ep)
        obs, info = env.reset(seed=ep_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        ep_best = env.env.best_seen_exec_time # unwrap ActionMasker
        per_episode_best.append(ep_best)
        if ep_best < overall_best_exec_time:
            overall_best_exec_time = ep_best
            overall_best_config = env.env.best_seen_config.copy()

    per_episode_best = np.array(per_episode_best)
    gaps = (per_episode_best - actual_best) / scale * 100
    overall_gap = (overall_best_exec_time - actual_best) / scale * 100

    return {
        "actual_best": actual_best,
        "median_exec_time": median_exec_time,
        "n_configs": n_configs,
        "baseline_mean": baseline["mean_gap"],
        "baseline_best": baseline["best_gap"],
        "agent_mean_gap": float(gaps.mean()),
        "agent_best_gap": float(overall_gap),
        "per_episode_best": per_episode_best,
        "overall_best_exec_time": overall_best_exec_time,
        "overall_best_config": overall_best_config,
    }


if __name__ == "__main__":
    cache_files, models_arg, agent, tag, n_episodes = parse_args()
    model_paths = resolve_models(models_arg, agent, tag, cache_files)

    if not model_paths:
        print("No models found to evaluate.")
        raise SystemExit(1)

    print(f"Using cache files: {[c.name for c in cache_files]}")
    print(f"Evaluating {len(model_paths)} model(s): {[m.name for m in model_paths]}")

    probe_backend = CacheBackend(str(cache_files[0]))
    RLEnv = KernelTunerRLEnvBig if agent == 'perparameter' else KernelTunerRLEnvSmall
    probe_env = RLEnv(probe_backend)
    n_params = probe_env.number_of_parameters
    base_dim = n_params + 3

    # per_model_gpu_mean[model_name][gpu] = agent_mean_gap
    per_model_gpu_mean = {}
    per_model_gpu_best = {}
    random_per_gpu = {}

    for model_path in model_paths:
        model = MaskablePPO.load(model_path)

        # Infer the obs format this model expects from its observation space.
        model_obs_dim = model.observation_space.shape[0]
        if model_obs_dim != base_dim:
            raise ValueError(
                f"{model_path.name}: obs dim {model_obs_dim} does not match expected dim {base_dim}. " 
                " Wrong kernel for this model?")

        print(f"  {model_path.name}: obs={model_obs_dim} ")

        per_model_gpu_mean[model_path.name] = {}
        per_model_gpu_best[model_path.name] = {}
        for cache_file in cache_files:
            res = evaluate_one(model, cache_file, n_episodes, agent)
            per_model_gpu_mean[model_path.name][cache_file.name] = res["agent_mean_gap"]
            per_model_gpu_best[model_path.name][cache_file.name] = res["agent_best_gap"]
            random_per_gpu[cache_file.name] = res["baseline_mean"]

    gpus = sorted(random_per_gpu.keys())
    
    print("\nRESULTS")
    print("-" * 50)
    
    model_overall = {}
    for model_path in model_paths:
        gaps = [per_model_gpu_mean[model_path.name][gpu] for gpu in gpus]
        avg = float(np.mean(gaps))
        model_overall[model_path.name] = avg
        
        print(f"\n{model_path.name}")
        for gpu in gpus:
            mean_gap = per_model_gpu_mean[model_path.name][gpu]
            random_gap = random_per_gpu[gpu]
            verdict = "✓" if mean_gap < random_gap else "✗"
            print(f"  {gpu:<30} gap: {mean_gap:>6.2f}%  random: {random_gap:>6.2f}%  {verdict}")
        print(f"  GPU average: {avg:.2f}%")
    
    best_model = min(model_overall, key=model_overall.get)
    print(f"\nBest model: {best_model}")
    print(f"Mean gap:   {model_overall[best_model]:.2f}%")
