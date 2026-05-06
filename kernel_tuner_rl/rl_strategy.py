import numpy as np
import gymnasium as gym
from kernel_tuner.searchspace import Searchspace

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


from kernel_tuner_rl.backends import KernelTunerBackend
from kernel_tuner_rl.env import KernelTunerRLEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


def tune(searchspace: Searchspace, runner, tuning_options):
    backend = KernelTunerBackend(searchspace, runner, tuning_options)
    env = KernelTunerRLEnv(backend)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load("models/ppo_kernel_tuner")

    seed = 42
    obs, info = env.reset(seed=seed)
    np.random.seed(seed)
    truncated = False
    terminated = False

    while not truncated and not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    return backend.runner_results
