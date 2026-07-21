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

    def _candidate_for_action(self, action):
        """Return the config this action leads to, or None if out of range."""
        param_index = action // 2
        direction = action % 2
        key = self.param_keys[param_index]
        values = self.tune_params[key]
        index = values.index(self.current_config[key])
 
        if direction == 0:
            if index <= 0:
                return None
            new_index = index - 1
        else:
            if index >= len(values) - 1:
                return None
            new_index = index + 1
 
        candidate = self.current_config.copy()
        candidate[key] = values[new_index]
        return candidate

    def action_masks(self) -> np.ndarray:
        """
        Create a boolean array for each action stating if it is valid or not

        Return:
            - boolean array showing valid and invalid actions as True and False respectively
        """
        candidates = []
        validity = []
        for action in range(self.action_space.n):
            candidate = self._candidate_for_action(action)
            candidates.append(candidate)
            validity.append(candidate is not None and self.backend.is_valid(candidate))
        return self._apply_novelty_mask(validity, candidates)

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

        new_config = self._candidate_for_action(action)
        # NOTE: with action masking, new_config is always valid; kept explicit
        # for check_env() compatibility.
        self.prev_exec_time = self.current_exec_time
        self.current_config = new_config
        self.step_count += 1
 
        truncated = self.step_count >= self.max_steps
 
        self._update_after_step()
 
        if self.backend.has_precomputed_stats:
            terminated = self.current_exec_time == self.optimum_exec_time
        else:
            terminated = False
 
        reward = self._compute_reward(terminated)
 
        observation = self._get_obs()
        info = self._get_info()
        info["terminated_at_optimum"] = terminated
 
        return observation, reward, terminated, truncated, info
