import json
import gzip
from pathlib import Path


def load_json(file):
    opener = gzip.open if str(file).endswith(".gz") else open
    with opener(file, "rt") as f:
        return json.load(f)


def cache_to_searchSpaceInfo(cache_file):
    # Parse the json into a python dict
    parsed_cache = load_json(cache_file)
    if "tune_params" not in parsed_cache:
        raise ValueError(f"Unsupported cache format. Found keys: {list(parsed_cache.keys())}")

    # gather needed info from the json cache
    tune_params = parsed_cache["tune_params"]

    cache_with_exec_time = parsed_cache["cache"]

    keys = list(tune_params.keys())
    searchspace = [dict(zip(keys, map(eval, key.split(",")))) for key in cache_with_exec_time.keys()]
    searchspace = [
        config
        for config in searchspace
        if isinstance(cache_with_exec_time[",".join(str(v) for v in config.values())]["time"], (int, float))
        and cache_with_exec_time[",".join(str(v) for v in config.values())]["time"] != float("inf")
    ]
    return (
        searchspace,
        cache_with_exec_time,
        tune_params,
    )


def config_to_string(config):
    values = config.values() if isinstance(config, dict) else config
    return ",".join(str(v) for v in values)

def get_kernel_name(paths):
    if len(paths) == 1:
        return paths[0].parent.name
    
    parent_name = paths[0].parent.name

    for path in paths[1:]:
        if parent_name != path.parent.name:
            raise ValueError("Cache files contain different kernels")
        
    return parent_name
