import gymnasium as gym
import numpy as np

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from kernel_tuner_rl_env import KernelTunerRLEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


env = KernelTunerRLEnv("cache/convolution_milo/A4000.json.gz")
env = ActionMasker(env, mask_fn)

model = MaskablePPO.load("models/ppo_kernel_tuner")

obs, info = env.reset()
truncated = False
best_exec_time = float("inf")
best_config = None

while not truncated:
    action, _ = model.predict(obs, action_masks=env.action_masks())
    obs, reward, terminated, truncated, info = env.step(action)

    if info["best_seen_exec_time"] < best_exec_time:
        best_exec_time = info["best_seen_exec_time"]
        best_config = env.env.current_config

actual_best = min(env.env.searchspaceInfo._get_config_exec_time(c) for c in env.env.searchspaceInfo.searchspace)

print(f"Agent Best: {best_exec_time:.4f}ms")
print(f"Cache best: {actual_best:.4f}ms")
print(f"Gap: {((best_exec_time- actual_best) / actual_best * 100):.1f}%")
