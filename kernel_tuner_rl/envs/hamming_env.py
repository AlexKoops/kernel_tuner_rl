import gymnasium as gym
import numpy as np
from kernel_tuner_rl.envs import KernelTunerRLEnvBase

class KernelTunerRLEnvHamming(KernelTunerRLEnvBase):
    """
    Environment to train PPO-based agents with a Hamming-1 distance action space
    """
    def __init__(self, backend):
        super().__init__(backend)

        # The observation or state space for this environment is defined as follows:
        # - current kernel configuration
        # - current kernel configuration exec time
        # - best seen exec time
        # - step progress = step_count/max_steps
        # - the high is set to 1.0 since the parameter values and exec time will be normalized
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(self.number_of_parameters + 3,), dtype=np.float32
        )

        # for the action space we use single incremental steps,
        # this means that for only one parameter at the time, we increment or decrement the value
        # this gives us 2 moves per parameter, so a total of 2*parameters
        self.action_space = gym.spaces.Discrete(self.number_of_parameters * 2)

    def _get_obs(self):
        """
        Create a list of normalized config, execution times and step progress

        Return:
            - array of normalized observations
        """
        normalized_list = []
        # Current config
        normalized_list.extend(self._normalize_config(self.current_config))
        # Current exec time
        normalized_list.append(self._normalize_exec_time(self.current_exec_time))
        # Best seen exec time
        normalized_list.append(self._normalize_exec_time(self.best_seen_exec_time))
        # Step progress
        normalized_list.append(min(self.step_count / self.max_steps, 1.0))

        return np.array(normalized_list, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """
        Create a boolean array for each action stating if it is valid or not

        Return:
            - boolean array showing valid and invalid actions as True and False respectively
        """
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Move agent to next config, following the correct direction

        Parameters:
            - action: the chosen action to take by the agent

        Return:
            - observation: the new observation the agent sees after taking action
            - reward: the reward this action resulted in 
            - terminated: boolean if the agent used all its episode budget
            - truncated: boolean if the agent has found the optimal solution 
            - info: additional debug info
        """

        param_index = action // 2  # the parameter where the step will occur
        direction = action % 2  # the direction the parameter value will step to

        key_to_update = self.param_keys[param_index]
        new_config = self.current_config.copy()

        # get the index for the current parameter setting
        tune_params = self.tune_params
        tune_param_index = tune_params[key_to_update].index(new_config[key_to_update])
        self.prev_exec_time = self.current_exec_time

        # create new config based on the parameter value the agent decided to change
        # NOTE: to make the check_env() function pass, config validation checks should be added, since check_env does not use achtion masking
        if direction == 0:
            new_config[key_to_update] = tune_params[key_to_update][tune_param_index - 1]
        else:
            new_config[key_to_update] = tune_params[key_to_update][tune_param_index + 1]

        # set new config
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
        # return observation, reward, terminated, truncated and info

        return observation, reward, terminated, truncated, info
