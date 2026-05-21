from abc import ABC, abstractmethod
from typing import Optional
import random

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.runners.runner import Runner

from kernel_tuner_rl.utils import cache_to_searchSpaceInfo, config_to_string


class Backend(ABC):

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
    def get_random_start(self):
        pass

    @abstractmethod
    def get_max_exec(self):
        pass

    @abstractmethod
    def get_opti_exec(self):
        pass


class CacheBackend(Backend):

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
        return float(max(t for t in (self.get_exec_time(c) for c in self.searchspace)))

    def get_opti_exec(self):
        return float(min(t for t in (self.get_exec_time(c) for c in self.searchspace)))


class KernelTunerBackend(Backend):

    def __init__(self, searchspace_object: Searchspace, runner: Runner, tune_options, calibration_samples: int = 30):
        # {(16, 1): 0, (16, 2): 1, (32, 1): 2}
        # maps the configs to its index in self.list for fast lookup
        self.searchspace = searchspace_object

        # {"block_size_x": [16, 32, 64], "tile_size": [1, 2, 4]}
        self.tune_params = searchspace_object.tune_params

        # ["block_size_x", "tile_size"]
        self.param_names = searchspace_object.param_names

        # set of config tuples
        self.valid_configs = set(searchspace_object.list)

        self.runner = runner
        self.tune_options = tune_options
        # set to hold runner results
        self.runner_results = []

        self._calibrate(calibration_samples)

    def _calibrate(self, n_samples: int):
        valid_list = list(self.valid_configs)
        n = min(n_samples, len(valid_list))
        sample = random.sample(valid_list, n)

        times = []
        for config_tuple in sample:
            config = dict(zip(self.param_names, config_tuple))
            t = self.get_exec_time(config)
            if isinstance(t, float) and t != float("inf"):
                times.append(t)

        if not times:
            self._calibrated_min = 0.0
            self._calibrated_max = float("inf")
            self._has_stats = False
        else:
            self._calibrated_min = min(times)
            self._calibrated_max = max(times)
            self._has_stats = True

    @property
    def has_precomputed_stats(self):
        return self._has_stats

    def get_max_exec(self):
        return self._calibrated_max

    def get_opti_exec(self):
        return self._calibrated_min

    def get_tune_params_dict(self):
        return self.tune_params

    def get_searchspace_size(self):
        return len(self.valid_configs)

    # config: dict of tune parameter configurations
    #
    # output: exec time as a float
    def get_exec_time(self, config):
        config_tuple = tuple(list(config.values()))

        result = self.runner.run([config_tuple], self.tune_options)

        exec_time = result[0][self.tune_options.objective]
        if not isinstance(exec_time, float):
            return float("inf")

        self.runner_results.append(result[0])
        return exec_time

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

