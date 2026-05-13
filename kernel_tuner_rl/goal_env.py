import gymnasium as gym
from typing import Optional
import numpy as np
from kernel_tuner_rl.base_env import KernelTunerRLEnvBase

class KernelTunerRLEnvGoal(KernelTunerRLEnvBase):
    def __init__(self, backend, speedup):
        super().__init__(backend)

        self.deployment_desired_speedup = speedup
        self.goal_tolerance = 1e-4

        self.desired_goal = None

        inner_obs_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.number_of_parameters + 2,),
            dtype=np.float32,
        )

        goal_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict({
            "observation":   inner_obs_space,
            "achieved_goal": goal_space,
            "desired_goal":  goal_space,
        })

        self.max_step_size = 3
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.number_of_parameters,), dtype=np.float32,
        )

    def _speedup(self, exec_time):
        if exec_time <= 0 or exec_time == float('inf'):
            return 0.0
        return float(self.episode_start_exec_time / exec_time)

    def _compute_desired_goal(self):
        if self.backend.has_precomputed_stats:
            target_speedup = self.episode_start_exec_time / self.optimum_exec_time
        else:
            target_speedup = self.deployment_desired_speedup
        return np.array([target_speedup], dtype=np.float32)

    def _get_inner_obs(self):
        normalized_list = []
        normalized_list.extend(self._normalize_config(self.current_config))
        normalized_list.append(self._normalize_exec_time(self.current_exec_time))
        normalized_list.append(min(self.step_count / self.max_steps, 1.0))

        return np.array(normalized_list, dtype=np.float32)

    def _get_obs(self):
        return {
            "observation": self._get_inner_obs(),
            "achieved_goal": np.array([self._speedup(self.current_exec_time)], dtype=np.float32),
            "desired_goal": self.desired_goal.copy()
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved = np.asarray(achieved_goal, dtype=np.float32).reshape(-1)
        desired = np.asarray(desired_goal, dtype=np.float32).reshape(-1)
        reached = achieved >= (desired - self.goal_tolerance)

        return np.where(reached, 0.0, -1.0).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self.desired_goal = self._compute_desired_goal()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self.prev_exec_time = self.current_exec_time

        new_config = self.current_config.copy()
        for i, key in enumerate(self.param_keys):
            values = self.tune_params[key]
            if len(values) <= 1:
                continue

            current_idx = values.index(new_config[key])
            step_delta = int(round(float(action[i]) * self.max_step_size))
            new_idx = int(np.clip(current_idx + step_delta, 0, len(values) - 1))
            new_config[key] = values[new_idx]

        if not self.backend.is_valid(new_config):
            new_config = self.current_config.copy()

        self.current_config = new_config
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        self.current_exec_time = self.backend.get_exec_time(self.current_config)
        if self.current_exec_time < self.best_seen_exec_time:
            self.best_seen_exec_time = self.current_exec_time
            self.best_seen_config = self.current_config.copy()

        observation = self._get_obs()
        achieved = observation["achieved_goal"]
        terminated = bool(achieved[0] >= self.desired_goal[0] - self.goal_tolerance)

        info = self._get_info()
        info["terminated_at_optimum"] = terminated and self.backend.has_precomputed_stats

        reward_arr = self.compute_reward(achieved, self.desired_goal, info)
        reward = float(reward_arr.item())

        return observation, reward, terminated, truncated, info

