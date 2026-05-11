from typing import Optional
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import random


class KernelTunerRLEnv(gym.Env):
    def __init__(self, backend):
        super().__init__()

        self.backend = backend

        self.current_config = None
        self.step_count = None
        self.current_exec_time = None
        self.prev_exec_time = None
        self.gamma = 0.999

        # percentage of searchspace we will use as a cutoff point in training
        searchspace_percentage = 0.1

        # collect parameter info
        self.tune_params = backend.get_tune_params_dict()
        self.param_keys = list(self.tune_params.keys())
        self.number_of_parameters = len(self.param_keys)

        # for the max steps we define a set amount, it is set to a percentage of the searchspace size

        self.max_steps = backend.get_searchspace_size() * searchspace_percentage
        # self.max_steps = 100

        if backend.has_precomputed_stats:
            # highest config execution time, will be used for the normalization
            self.max_exec_time = backend.get_max_exec()
            # the optimum exectution time, will be used to calculate rewards
            self.optimum_exec_time = backend.get_opti_exec()

            print(f"max_exec_time: {self.max_exec_time}")
            print(f"optimum_exec_time: {self.optimum_exec_time}")
        else:
            self.max_exec_time = float("inf")
            self.optimum_exec_time = 0.0
            print("Backend has no precomputed stats")

        self.best_seen_exec_time = self.current_exec_time

        self.best_seen_config = None

        # The observation or state space for this environment is defined as follows:
        #   * current kernel configuration
        #   * current kernel configuration exec time
        #   * step progress = step_count/max_steps
        #   * the high is set to 1.0 since the parameter values and exec time will be normalized
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(self.number_of_parameters + 2,), dtype=np.float32
        )
        print(f"number of parameters: {self.number_of_parameters}")

        # for the action space we use single incremental steps,
        # this means that for only one parameter at the time, we increment or decrement the value
        # this gives us 2 moves per parameter, so a total of 2*parameters
        self.action_space = gym.spaces.Discrete(self.number_of_parameters * 2)

    def action_masks(self) -> np.ndarray:
        mask_list = []
        tune_params = self.tune_params
        for key in self.current_config:
            index = tune_params[key].index(self.current_config[key])

            if index > 0:
                # Create neighbour to check if it is not a restricted config
                neighbour = self.current_config.copy()
                neighbour[key] = tune_params[key][index - 1]
                mask_list.append(self.backend.is_valid(neighbour))
            else:
                mask_list.append(False)

            if index < len(tune_params[key]) - 1:
                # Create neighbour to check if it is not a restricted config
                neighbour = self.current_config.copy()
                neighbour[key] = tune_params[key][index + 1]
                mask_list.append(self.backend.is_valid(neighbour))
            else:
                mask_list.append(False)

        return np.array(mask_list, dtype=np.bool_)

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

        return min(self.best_seen_exec_time / exec_time, 1.0)

    def _get_obs(self):
        # creates a normalized nparray that represents the observation
        normalized_list = []

        # add current config and best seen config
        normalized_list.extend(self._normalize_config(self.current_config))

        # add the execution times
        current_exec_time = self.current_exec_time
        normalized_list.append(self._normalize_exec_time(current_exec_time))

        # add the point in time, based on the step_count
        normalized_list.append(min(self.step_count / self.max_steps, 1.0))

        return np.array(normalized_list, dtype=np.float32)

    def _get_info(self):
        # Generate extra info for debugging purposes
        return {"best_seen_exec_time": self.best_seen_exec_time}

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

        return self._get_obs(), self._get_info()

    def step(self, action):
        # move agent to next config, follpwing the correct direction

        param_index = action // 2  # the parameter where the step will occur
        direction = action % 2  # the direction the parameter value will step to

        key_to_update = self.param_keys[param_index]

        new_config = self.current_config.copy()

        # get the index for the current parameter setting
        tune_params = self.tune_params
        tune_param_index = tune_params[key_to_update].index(new_config[key_to_update])
        self.prev_exec_time = self.current_exec_time

        # first 2 if elif statements are only added for check_env() to be able to pass, since this does not support action maskign
        if direction == 0 and tune_param_index == 0:
            # can't step down, stay in place
            new_config = self.current_config.copy()
        elif direction == 1 and tune_param_index == len(tune_params[key_to_update]) - 1:
            # can't step up, stay in place
            new_config = self.current_config.copy()

        # these 2 are the only ifs that should trigger when action masking class is used
        elif direction == 0:
            new_config[key_to_update] = tune_params[key_to_update][tune_param_index - 1]
        else:
            new_config[key_to_update] = tune_params[key_to_update][tune_param_index + 1]

        # set new config
        self.current_config = new_config

        self.step_count += 1

        # check if truncated
        if self.step_count >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # get reward, based on current and new exec_time

        self.current_exec_time = self.backend.get_exec_time(self.current_config)
        if self.current_exec_time < self.best_seen_exec_time:
            self.best_seen_exec_time = self.current_exec_time

            self.best_seen_config = self.current_config.copy()

        if self.backend.has_precomputed_stats:
            terminated = self.current_exec_time == self.optimum_exec_time
        else:
            terminated = False

        # Potential-Based Reward Shaping
        # Based on Ng at al (1999): Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping

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

        if np.isnan(reward):
            print(f"NaN reward! prev: {self.prev_exec_time}, current: {self.current_exec_time}")

        observation = self._get_obs()
        info = self._get_info()
        info["terminated_at_optimum"] = terminated
        # return observation, reward, terminated, truncated and info

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = KernelTunerRLEnv("cache/convolution_milo/A4000.json.gz")
    check_env(env)
