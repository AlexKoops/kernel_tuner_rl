from typing import Optional
import numpy as np
import gymnasium as gym

from kernel_tuner import Searchspace


class KernelTunerRLEnv(gym.Env):
    def __init__(self, Searchspace: Searchspace):
        super().__init__()

        self.searchspace = Searchspace

        # kernel_tuner.Searchspace class has a self.tune_params: dict parameter
        self.number_of_parameters = len(Searchspace.tune_params)

        # The observation or state space for this environment is defined as follows:
        #   * current kernel configuration
        #   * the performance of the current kernel configuration, hence the ( + 1) in the shape
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(self.number_of_parameters + 1,), dtype=np.float32
        )

        # for the action space we use single incremental steps,
        # this means that for only one parameter at the time, we increment or decrement the value
        # this gives us 2 moves per parameter, so a total of 2*parameters
        self.action_space = gym.spaces.Discrete(self.number_of_parameters * 2)

    def _get_obs(self):
        # Convert internal state to observation format

        return {"kernelConfiguration": self._agent_location}

    def _get_info(self):
        # Generate extra info for debugging purposes
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        pass

    def step(self, action):
        pass
