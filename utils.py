from itertools import product
import json
import gzip


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
    return (
        searchspace,
        cache_with_exec_time,
        tune_params,
    )
