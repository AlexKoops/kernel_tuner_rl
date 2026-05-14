import json
import gzip
import numpy as np
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


def random_baseline(backend, n_samples, n_trials=5, seed=0):
    """Best-of-n_samples random valid configs, averaged over n_trials.
    Returns dict with mean_gap, best_gap, std_gap."""
    rng = np.random.default_rng(seed)
    configs = backend.searchspace
    all_times = sorted(
        t for t in (backend.get_exec_time(c) for c in configs)
        if t != float("inf")
    )
    actual_best = all_times[0]
    scale = np.median(all_times) - actual_best

    gaps = []
    for trial in range(n_trials):
        idx = rng.choice(len(configs), size=min(n_samples, len(configs)), replace=False)
        times = [backend.get_exec_time(configs[i]) for i in idx]
        times = [t for t in times if t != float("inf")]
        best = min(times) if times else float("inf")
        gaps.append((best - actual_best) / scale * 100)

    gaps = np.array(gaps)
    return {"mean_gap": float(gaps.mean()), "best_gap": float(gaps.min()), "std_gap": float(gaps.std())}
