from abc import ABC, abstractmethod
from typing import Optional
import random

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
        self.valid_configs = {config_to_string(config) for config in self.searchspace}

    @property
    def has_precomputed_stats(self):
        return True

    def get_tune_params_dict(self):
        return self.tune_params

    def get_searchspace_size(self):
        return len(self.searchspace)

    def get_exec_time(self, config):
        cache_key = config_to_string(config)
        time = self.cache[cache_key]["time"]
        if isinstance(time, str):
            return float("inf")
        return time

    def is_valid(self, config):
        conf_str = config_to_string(config)
        return conf_str in self.valid_configs

    def get_random_start(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        return random.choice(self.searchspace)

    def get_max_exec(self):
        return float(max(t for t in (self.get_exec_time(c) for c in self.searchspace)))

    def get_opti_exec(self):
        return float(min(t for t in (self.get_exec_time(c) for c in self.searchspace)))
