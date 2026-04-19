import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from kernel_tuner_rl_env import KernelTunerRLEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


if __name__ == "__main__":
    env = KernelTunerRLEnv("clean_cache/convolution_milo/A4000.json.gz")
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.learn(total_timesteps=150_000)

    model.save("models/ppo_kernel_tuner")
