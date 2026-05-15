import numpy as np
import gymnasium as gym
from kernel_tuner.searchspace import Searchspace

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


from kernel_tuner_rl.backends import KernelTunerBackend
from kernel_tuner_rl.discrete_env import KernelTunerRLEnvDiscrete


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


def tune(searchspace: Searchspace, runner, tuning_options):
    backend = KernelTunerBackend(searchspace, runner, tuning_options)
    env = KernelTunerRLEnvDiscrete(backend)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO.load("models/ppo_convolution_milo")

    # Run multiple episodes and return the best, matching eval semantics.
    best_results = []
    best_time = float("inf")
    for ep in range(5):
        backend.runner_results = []  # reset before each episode
        obs, info = env.reset(seed=42 + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        if env.env.best_seen_exec_time < best_time:
            best_time = env.env.best_seen_exec_time
            best_results = list(backend.runner_results)

    return best_results
