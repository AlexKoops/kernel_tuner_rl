import gymnasium as gym
from typing import Optional
import numpy as np
from abc import abstractmethod

class KernelTunerRLEnvBase(gym.Env):
    """
    Shared base class for both environments, defines shared logic in the environments
    """
    def __init__(self, backend):
        super().__init__()

        self.backend = backend # cache or kernelTuner backend
        self.current_config = None 
        self.step_count = None # amount of steps agent has in episode
        self.current_exec_time = None
        self.prev_exec_time = None
        self.gamma = 0.999

        # parameters and info around it
        self.tune_params = backend.get_tune_params_dict()
        self.param_keys = list(self.tune_params.keys())
        self.number_of_parameters = len(self.param_keys)
        self.total_options = sum(len(options) for options in self.tune_params.values())

        self.max_steps = backend.get_max_steps() # max episode length, clipped at 1000

        # if precomputed stats exist, use these
        if backend.has_precomputed_stats:
            self.max_exec_time = backend.get_max_exec() # used for normalization
            self.optimum_exec_time = backend.get_opti_exec() # used for reward calculation

        else: 
            self.max_exec_time = float("inf")
            self.optimum_exec_time = 0.0

        self.best_seen_exec_time = None
        self.best_seen_config = None
        self.episode_start_exec_time = None

        self.visited_configs = set()

    def _config_key(self, config):
        """Hashable key for a configuration dict."""
        return tuple(config[k] for k in self.param_keys)
 
    def _mark_visited(self, config):
        self.visited_configs.add(self._config_key(config))
 
    def _is_unvisited(self, config):
        return self._config_key(config) not in self.visited_configs
 
    def _apply_novelty_mask(self, validity_mask, candidate_configs):
        """
        Given a boolean validity mask and the candidate config for each action,
        prefer actions leading to unvisited configs. Falls back to the plain
        validity mask if no valid action leads anywhere new (prevents deadlock).
        """
        novelty_mask = np.array(
            [validity_mask[i] and self._is_unvisited(candidate_configs[i])
             if candidate_configs[i] is not None else False
             for i in range(len(validity_mask))],
            dtype=np.bool_,
        )
        if novelty_mask.any():
            return novelty_mask
        return np.asarray(validity_mask, dtype=np.bool_)



    def _get_info(self):
        return {
            "best_seen_exec_time": self.best_seen_exec_time,
            "unique_configs_visited": len(self.visited_configs),
        }

    def _normalize_config(self, config):
        """
        Normalizes the configuration based on the amount of options per parameter.

        Parameters:
            config: the config to normalize

        Return:
            list of normalized parameter values for the given config
        """
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

    # min-max normalization on the exec time
    def _normalize_exec_time(self, exec_time):
        """
        Min-Max normalization on the execution time

        Parameters:
            - exec_time: the execution time to normalize

        Return:
            - normalized execution time as a float
        """
        if exec_time == float("inf"):
            return 1.0
        if self.backend.has_precomputed_stats:
            diff = self.max_exec_time - self.optimum_exec_time
            if diff <= 0:
                return 0.0
            return float(np.clip(
                (exec_time - self.optimum_exec_time) / diff,
                0.0, 1.0,
            ))
        # fallback when no precomputed statistics exist
        return float(self.episode_start_exec_time / (self.episode_start_exec_time + exec_time))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset function for the environment. Places agent on new random configuration and resets class attributes
        """
        super().reset(seed=seed)

        # keep_visited is set to True during deployment to ensure restarts dont reset visited configurations
        keep_visited = bool(options.get("keep_visited", False)) if options else False
        if not keep_visited:
            self.visited_configs = set()

        self.current_config = self.backend.get_random_start(self.np_random) # place agent on a random config
        self.step_count = 0
        self.current_exec_time = self.backend.get_exec_time(self.current_config) # get first config exec time
        self.prev_exec_time = self.current_exec_time
        self.best_seen_exec_time = self.current_exec_time
        self.best_seen_config = self.current_config.copy()
        self.episode_start_exec_time = self.current_exec_time
        self._mark_visited(self.current_config)

    def _update_after_step(self):
        """
        Shared bookkeeping subclasses call inside step() after setting the new
        current_config: evaluates it, updates best-seen (remembering the
        previous best for the reward), and records the visit.
        """
        self.prev_best_seen_exec_time = self.best_seen_exec_time
        self.current_exec_time = self.backend.get_exec_time(self.current_config)
        if self.current_exec_time < self.best_seen_exec_time:
            self.best_seen_exec_time = self.current_exec_time
            self.best_seen_config = self.current_config.copy()
        self._mark_visited(self.current_config)

    def _compute_reward(self, terminated):
        """
        Function to determine the reward after each agent action in a state. Uses Potential-Based Reward shaping

        Parameters:
            - terminated: boolean if the agent has found the optimum or not

        Return:
            - reward: the calculated reward for this action in state.
        """
        if self.backend.has_precomputed_stats:
            # Reflect the objective directly
            true_reward = 1.0 if terminated else 0.0

            # add Potential-Based shaping to the reward
            phi_current = self.optimum_exec_time / self.best_seen_exec_time
            phi_prev = self.optimum_exec_time / self.best_seen_exec_time
            
            # gives a reward, based on distance between current config and optimum
            F = self.gamma * phi_current - phi_prev

            reward = true_reward + F
        else:
            reward = 0.0  # if no precomputed stats, the agent is deployed so no reward needed since no training

        return reward

            


    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
