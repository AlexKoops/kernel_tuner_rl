from typing import Optional
import numpy as np
import gymnasium as gym
import random

import searchspaceInfo


class KernelTunerRLEnv(gym.Env):
    def __init__(self, cache_file):
        super().__init__()

        self.current_config = None
        self.step_count = None
        self.initial_exec_time = None

        # percentage of searchspace we will use as a cutoff point in training
        searchspace_percentage = 20

        self.searchspaceInfo = searchspaceInfo(cache_file)

        # kernel_tuner.Searchspace class has a self.tune_params: dict parameter
        self.number_of_parameters = self.searchspaceInfo.number_of_parameters

        # for the max steps we define a set amount, it is set to a percentage of the searchspace size
        self.max_steps = len(self.searchspaceInfo.searchspace) // searchspace_percentage

        # highest config execution time, will be used for the normalization
        self.max_exec_time = max(
            self.searchspaceInfo._get_config_exec_time(c) for c in self.searchspaceInfo.searchspace
        )

        # The observation or state space for this environment is defined as follows:
        #   * current kernel configuration
        #   * the performance of the current kernel configuration, hence the ( + 1) in the shape
        #   * the high is set to 1.0 since the parameter values and exec time will be normalized
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(self.number_of_parameters + 1,), dtype=np.float32
        )

        # for the action space we use single incremental steps,
        # this means that for only one parameter at the time, we increment or decrement the value
        # this gives us 2 moves per parameter, so a total of 2*parameters
        self.action_space = gym.spaces.Discrete(self.number_of_parameters * 2)

    def _get_obs(self):
        # creates a normalized nparray that represents the observation
        normalized_list = []
        # create the normalized parameters first
        for key in self.current_config:
            index = self.searchspaceInfo.tune_params[key].index(self.current_config[key])
            tune_param_length = len(self.searchspaceInfo.tune_params[key])

            # prevent devision by zero
            if tune_param_length == 1:
                normalized_param = 0.0
            else:
                normalized_param = index / (tune_param_length - 1)

            normalized_list.append(normalized_param)

        # append the normalized exec time at the end
        exec_time = self.searchspaceInfo._get_config_exec_time(self.current_config)
        normalized_list.append(exec_time / self.max_exec_time)

        return np.array(normalized_list, dtype=np.float32)

    def _get_info(self):
        # Generate extra info for debugging purposes
        pass

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self.current_config = random.choice(self.searchspaceInfo.searchspace)
        self.step_count = 0
        self.initial_exec_time = self.searchspaceInfo._get_config_exec_time(self.current_config)

        return (self._get_obs(), {})

    def step(self, action):
        pass
