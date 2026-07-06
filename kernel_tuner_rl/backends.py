"""
Backend abstractions used by the environment to handle both cache file input and Kernel Tuner input

Contains two abstractions:
    - CacheBackend: handles cachefile input
    - KernelTunerBackend: handles inputs from Kernel Tuner
"""
from abc import ABC, abstractmethod
import numpy as np
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.runners.runner import Runner
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner.util import StopCriterionReached


from kernel_tuner_rl.utils import cache_to_searchSpaceInfo, config_to_string


class Backend(ABC):
    """
    Shared base for both backend abstractions
    """

    @property
    @abstractmethod
    def has_precomputed_stats(self):
        pass

    @abstractmethod
    def get_tune_params_dict(self):
        pass

    @abstractmethod
    def get_searchspace_size(self):
        pass

    @abstractmethod
    def get_exec_time(self, config):
        pass

    @abstractmethod
    def is_valid(self, config):
        pass

    @abstractmethod
    def get_random_start(self, rng):
        pass

    @abstractmethod
    def get_max_exec(self):
        pass

    @abstractmethod
    def get_opti_exec(self):
        pass

    def get_max_steps(self):
        """
        Sets the episode length to 5% of the searchspace size, capped at 1000 timesteps to avoid extremely long episodes
        """
        searchspace_percentage = 0.05
        return int(min(1000, self.get_searchspace_size() * searchspace_percentage))


class CacheBackend(Backend):
    """
    Backend abstraction handling a cache file as data source.
    """

    def __init__(self, cache_file):
        # searchspace has form:
        # [
        #     {"block_size": 16, "tile_size": 1},
        #     {"block_size": 16, "tile_size": 2},
        #     {"block_size": 32, "tile_size": 2},
        # ]

        # cache has form:
        # {
        #     "16,1": {"time": 0.023},
        #     "16,2": {"time": 0.019},
        #     "32,2": {"time": 0.031},
        # }

        # tune_params has form:
        # {
        #     "block_size": [16, 32, 64],
        #     "tile_size": [1, 2, 4]
        # }
        self.searchspace, self.cache, self.tune_params = cache_to_searchSpaceInfo(cache_file)

        # valid_configs has form:
        # {
        #     "16,1",
        #     "16,2"
        # }
        self.valid_configs = {config_to_string(c) for c in self.searchspace}
        self.all_times = list(t for t in (self.get_exec_time(c) for c in self.searchspace))
        self.valid_times = [t for t in self.all_times if t != float('inf')]

    @property
    def has_precomputed_stats(self):
        return True

    def get_tune_params_dict(self):
        return self.tune_params

    def get_searchspace_size(self):
        return len(self.searchspace)

    # config: dict of tune parameter configurations
    #
    # output: exec time as a float
    def get_exec_time(self, config):
        cache_key = config_to_string(config)
        time = self.cache[cache_key]["time"]
        if isinstance(time, str):
            return float("inf")
        return time

    # config: dict of tune parameter configurations
    #
    # output: bool
    def is_valid(self, config):
        config_str = config_to_string(config)
        return config_str in self.valid_configs

    # output: dict of parameter options and their configs
    # example: {"block_size": 16, "tile_size": 1}
    def get_random_start(self, rng):
        index = rng.integers(0, len(self.searchspace))
        return self.searchspace[index]

    def get_max_exec(self):
        return float(max(self.valid_times))

    def get_opti_exec(self):
        return float(min(self.valid_times))

    def get_median_exec(self):
        return float(np.median(self.valid_times))


class KernelTunerBackend(Backend):
    """
    Backend handling Kernel Tuner input as a data source
    """

    def __init__(self, searchspace_object: Searchspace, runner: Runner, tune_options, calibration_samples: int = 5):
        # {(16, 1): 0, (16, 2): 1, (32, 1): 2}
        # maps the configs to its index in self.list for fast lookup
        self.searchspace = searchspace_object

        # {"block_size_x": [16, 32, 64], "tile_size": [1, 2, 4]}
        self.tune_params = searchspace_object.tune_params

        # ["block_size_x", "tile_size"]
        self.param_names = searchspace_object.param_names

        # set of config tuples
        self.valid_configs = set(searchspace_object.list)

        self.tune_options = tune_options
        
        self.cost_func = CostFunc(searchspace_object, tune_options, runner)

        self._calibrate(max(calibration_samples, 5))

    def _calibrate(self, n_samples):
        """
        Gathers some environment data when there don't exist global stats of the environment.
        This data is used for normalization during deployment, data is set as class attributes

        Parameters:
            n_samples: amount of samples used to calibrate the global stats with
        """
        valid_list = list(self.valid_configs)
        rng = np.random.default_rng()
        n = min(n_samples, len(valid_list))
        choices = rng.choice(len(valid_list), size=n, replace=False)
        sample = [valid_list[c] for c in choices]

        times = []
        try:
            for config in sample:
                t = self.cost_func(config, check_restrictions=False)
                if isinstance(t, float) and t != float("inf"):
                    times.append(t)
        except StopCriterionReached:
            pass  # budget ran out during calibration

        if len(times) < 2:
            self._calibrated_min = 0.0
            self._calibrated_max = float("inf")
            self._calibrated_median = float("inf")
            self._has_stats = False
        else:
            self._calibrated_min = float(min(times))
            self._calibrated_max = float(max(times))
            self._calibrated_median = float(np.median(times))
            self._has_stats = True

    @property
    def has_precomputed_stats(self):
        return self._has_stats

    def get_max_exec(self):
        return self._calibrated_max

    def get_opti_exec(self):
        return self._calibrated_min

    def get_median_exec(self):
        return self._calibrated_median

    def get_tune_params_dict(self):
        return self.tune_params

    def get_searchspace_size(self):
        return len(self.valid_configs)

    # config: dict of tune parameter configurations
    #
    # output: exec time as a float
    def get_exec_time(self, config):
        config_tuple = tuple(config[k] for k in self.param_names)
        t = self.cost_func(config_tuple, check_restrictions=False)
        return t if isinstance(t, float) else float("inf")

    # config: dict of tune parameter configurations
    #
    # output: bool
    def is_valid(self, config):
        config_tuple = tuple(config[k] for k in self.param_names)
        return config_tuple in self.valid_configs

    # output: dict of parameter options and their configs
    # example: {"block_size": 16, "tile_size": 1}
    def get_random_start(self, rng):
        valid_list = list(self.valid_configs)
        index = rng.integers(0, len(valid_list))
        choice = valid_list[index]
        return dict(zip(self.param_names, choice))

