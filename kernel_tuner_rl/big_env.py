import gymnasium as gym
import numpy as np
from kernel_tuner_rl.base_env import KernelTunerRLEnvBase


class KernelTunerRLEnvBig(KernelTunerRLEnvBase):
    def __init__(self, backend):
        super().__init__(backend)

        # Observation space is identical to the discrete env's lean obs.
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0,
            shape=(self.number_of_parameters + 3,),
            dtype=np.float32,
        )

        # Action space: one slot per (parameter, value-index) pair.
        self.action_space = gym.spaces.Discrete(self.total_options)

        # Lookup table from action index -> (parameter_key, value_index)
        self._action_lookup = []
        for key in self.param_keys:
            for v_idx in range(len(self.tune_params[key])):
                self._action_lookup.append((key, v_idx))

    def _get_obs(self):
        normalized_list = []
        normalized_list.extend(self._normalize_config(self.current_config))
        normalized_list.append(self._normalize_exec_time(self.current_exec_time))
        normalized_list.append(self._normalize_exec_time(self.best_seen_exec_time))
        normalized_list.append(min(self.step_count / self.max_steps, 1.0))
        return np.array(normalized_list, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.total_options, dtype=np.bool_)
        for action_idx, (key, v_idx) in enumerate(self._action_lookup):
            current_v_idx = self.tune_params[key].index(self.current_config[key])
            if v_idx == current_v_idx:
                continue

            candidate = self.current_config.copy()
            candidate[key] = self.tune_params[key][v_idx]
            if self.backend.is_valid(candidate):
                mask[action_idx] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Decode the action into (parameter, target value).
        key_to_update, v_idx = self._action_lookup[action]
        new_value = self.tune_params[key_to_update][v_idx]

        self.prev_exec_time = self.current_exec_time

        # Apply the jump: set one parameter to its v_idx-th value.
        new_config = self.current_config.copy()
        new_config[key_to_update] = new_value
        self.current_config = new_config

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        self.current_exec_time = self.backend.get_exec_time(self.current_config)
        if self.current_exec_time < self.best_seen_exec_time:
            self.best_seen_exec_time = self.current_exec_time
            self.best_seen_config = self.current_config.copy()

        if self.backend.has_precomputed_stats:
            terminated = self.current_exec_time == self.optimum_exec_time
        else:
            terminated = False

        reward = self._compute_reward(terminated)

        observation = self._get_obs()
        info = self._get_info()
        info["terminated_at_optimum"] = terminated

        return observation, reward, terminated, truncated, info
