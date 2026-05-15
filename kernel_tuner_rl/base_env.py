import gymnasium as gym
from typing import Optional
import numpy as np
import random
from abc import abstractmethod

class KernelTunerRLEnvBase(gym.Env):
    def __init__(self, backend):
        super().__init__()

        self.backend = backend

        self.current_config = None
        self.step_count = None
        self.current_exec_time = None
        self.prev_exec_time = None
        self.gamma = 0.999

        searchspace_percentage = 0.1

        self.tune_params = backend.get_tune_params_dict()
        self.param_keys = list(self.tune_params.keys())
        self.number_of_parameters = len(self.param_keys)

        self.max_steps = int(min(3000, backend.get_searchspace_size() * searchspace_percentage))

        if backend.has_precomputed_stats:
            # highest config execution time, will be used for the normalization
            self.max_exec_time = backend.get_max_exec()
            # the optimum exectution time, will be used to calculate rewards
            self.optimum_exec_time = backend.get_opti_exec()

        else:
            self.max_exec_time = float("inf")
            self.optimum_exec_time = 0.0

        self.best_seen_exec_time = None

        self.best_seen_config = None

        self.episode_start_exec_time = None

        self.steps_since_improvement = None

        
    def _get_info(self):
        # Generate extra info for debugging purposes
        return {"best_seen_exec_time": self.best_seen_exec_time}

    def _normalize_config(self, config):
        normalized = []
        tune_params = self.tune_params
        for key in config:
            index = tune_params[key].index(config[key])
            tune_param_length = len(tune_params[key])
            if tune_param_length == 1:
                normalized.append(0.0)
            else:
                normalized.append(index / (tune_param_length - 1))
        return normalized

    def _normalize_exec_time(self, exec_time):
        if exec_time == float("inf"):
            return 1.0
        if self.backend.has_precomputed_stats:
            span = self.max_exec_time - self.optimum_exec_time
            if span <= 0:
                return 0.0
            return float(np.clip(
                (exec_time - self.optimum_exec_time) / span,
                0.0, 1.0,
            ))
        # Deployment fallback (no precomputed stats): keep the old behavior
        return float(self.episode_start_exec_time / (self.episode_start_exec_time + exec_time))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self.current_config = self.backend.get_random_start(seed)
        self.step_count = 0
        self.current_exec_time = self.backend.get_exec_time(self.current_config)
        self.prev_exec_time = self.current_exec_time

        self.best_seen_exec_time = self.current_exec_time
        self.best_seen_config = self.current_config.copy()

        self.steps_since_improvement = 0

        self.episode_start_exec_time = self.current_exec_time

    def _compute_reward(self, terminated):
        if self.backend.has_precomputed_stats:
            # Reflect the objective directly
            R_true = 1.0 if terminated else 0.0

            # add Potential-Based shaping to the reward
            phi_current = self.optimum_exec_time / self.current_exec_time
            phi_prev = self.optimum_exec_time / self.prev_exec_time
            shaping = self.gamma * phi_current - phi_prev

            reward = R_true + shaping
        else:
            reward = 0.0  # if no precomputed stats, the agent is deployed so no reward needed since no training

        return reward

            


    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
