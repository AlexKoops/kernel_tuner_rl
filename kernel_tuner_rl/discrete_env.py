import gymnasium as gym
import numpy as np
from kernel_tuner_rl.base_env import KernelTunerRLEnvBase

class KernelTunerRLEnvDiscrete(KernelTunerRLEnvBase):
    def __init__(self, backend):
        super().__init__(backend)

        # The observation or state space for this environment is defined as follows:
        #   * current kernel configuration
        #   * current kernel configuration exec time
        #   * step progress = step_count/max_steps
        #   * the high is set to 1.0 since the parameter values and exec time will be normalized
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(self.number_of_parameters * 2 + 4,), dtype=np.float32
        )

        # for the action space we use single incremental steps,
        # this means that for only one parameter at the time, we increment or decrement the value
        # this gives us 2 moves per parameter, so a total of 2*parameters
        self.action_space = gym.spaces.Discrete(self.number_of_parameters * 2)

    def _get_obs(self):
        normalized_list = []
        # Current config
        normalized_list.extend(self._normalize_config(self.current_config))
        # Best-seen config
        normalized_list.extend(self._normalize_config(self.best_seen_config))
        # Current exec time
        normalized_list.append(self._normalize_exec_time(self.current_exec_time))
        # Best-seen exec time
        normalized_list.append(self._normalize_exec_time(self.best_seen_exec_time))
        # Step progress
        normalized_list.append(min(self.step_count / self.max_steps, 1.0))
        # Steps since improvement
        normalized_list.append(min(self.steps_since_improvement / self.max_steps, 1.0))

        return np.array(normalized_list, dtype=np.float32)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
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
        # if direction == 0 and tune_param_index == 0:
        #     # can't step down, stay in place
        #     new_config = self.current_config.copy()
        # elif direction == 1 and tune_param_index == len(tune_params[key_to_update]) - 1:
        #     # can't step up, stay in place
        #     new_config = self.current_config.copy()

        # these 2 are the only ifs that should trigger when action masking class is used
        if direction == 0:
            new_config[key_to_update] = tune_params[key_to_update][tune_param_index - 1]
        else:
            new_config[key_to_update] = tune_params[key_to_update][tune_param_index + 1]

        # set new config
        self.current_config = new_config

        self.step_count += 1

        truncated = True if self.step_count >= self.max_steps else False

        self.current_exec_time = self.backend.get_exec_time(self.current_config)
        if self.current_exec_time < self.best_seen_exec_time:
            self.best_seen_exec_time = self.current_exec_time
            self.best_seen_config = self.current_config.copy()
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1


        if self.backend.has_precomputed_stats:
            terminated = self.current_exec_time == self.optimum_exec_time
        else:
            terminated = False

        reward = self._compute_reward(terminated)

        if np.isnan(reward):
            print(f"NaN reward! prev: {self.prev_exec_time}, current: {self.current_exec_time}")

        observation = self._get_obs()
        info = self._get_info()
        info["terminated_at_optimum"] = terminated
        # return observation, reward, terminated, truncated and info

        return observation, reward, terminated, truncated, info

