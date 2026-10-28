import os
import numpy as np
import gymnasium as gym
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import StopCriterionReached
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from kernel_tuner_rl.backends import KernelTunerBackend
from kernel_tuner_rl.envs import KernelTunerRLEnvPerparameter


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()


STALL_LIMIT = 15


def tune(searchspace: Searchspace, runner, tuning_options):
    run_rng = np.random.default_rng()
    base_seed = int(run_rng.integers(0, 2**31 - 1))

    backend = KernelTunerBackend(searchspace, runner, tuning_options)
    env = KernelTunerRLEnvPerparameter(backend)
    env = ActionMasker(env, mask_fn)

    MODEL_PATHS = {
        "convolution_kernel": "models/convolution_milo/perparameter_ppo_seed0",
        "dedispersion_kernel": "models/dedispersion_milo/perparameter_ppo_seed0",
        "Xgemm": "models/gemm_milo/perparameter_ppo_seed7",
        "calculate_temp": "models/hotspot_milo/perparameter_ppo_seed5",
    }
    kernel_name = runner.kernel_options.kernel_name

    model_path = MODEL_PATHS.get(kernel_name)
    if model_path is None:
        raise ValueError(f"{kernel_name} is not supported, supported kernels are: {list(MODEL_PATHS)}")

    # TODO: add path to models
    _rl_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ""))
    model = MaskablePPO.load(os.path.join(_rl_root, model_path))

    global_best_time = float("inf")
    restart_counter = 0
    steps_since_improvement = 0
    best_time_this_segment = float("inf")

    try:
        obs, info = env.reset(seed=base_seed)
        while True:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            current_best = env.env.best_seen_exec_time
            if current_best < best_time_this_segment:
                best_time_this_segment = current_best
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            if current_best < global_best_time:
                global_best_time = current_best

            if terminated:
                break

            if steps_since_improvement >= STALL_LIMIT:
                restart_counter += 1
                obs, info = env.reset(seed=base_seed + restart_counter)
                steps_since_improvement = 0
                best_time_this_segment = float("inf")

    except StopCriterionReached:
        pass

    return backend.cost_func.results
