from utils import cache_to_searchSpaceInfo


class SearchspaceInfo:
    def __init__(self, cache_file):
        # seachspace is the cartesian product of all tune_param options
        # cache is a dictionary with all configurations and its exec time
        # tune_params is a dict of all tuneable parameters paired with a list of valid values
        self.searchspace, self.cache, self.tune_params = cache_to_searchSpaceInfo(cache_file)
        self.number_of_parameters = len(self.tune_params)
        self.neighbours = {}

        self._compute_neighbours()

    def _compute_neighbours(self):
        tune_params = self.tune_params
        neighbours = {}
        for config in self.searchspace:
            local_neighbours = []
            for key in config:  # loop over all configs
                # we get the index in the list with paramete values for param i
                index = tune_params[key].index(config[key])
                if index > 0:  # there is a valid parameter option smaller then current
                    neighbour = config.copy()
                    neighbour[key] = tune_params[key][index - 1]
                    neighbour_key = ",".join(str(v) for v in neighbour.values())
                    if neighbour_key in self.cache:
                        local_neighbours.append(neighbour)  # there is a valid parameter option bigger then current
                if index < len(tune_params[key]) - 1:
                    neighbour = config.copy()
                    neighbour[key] = tune_params[key][index + 1]
                    neighbour_key = ",".join(str(v) for v in neighbour.values())
                    if neighbour_key in self.cache:
                        local_neighbours.append(neighbour)

            neighbours[tuple(config.values())] = local_neighbours

        self.neighbours = neighbours

        empty = [k for k, v in neighbours.items() if len(v) == 0]
        print(f"Configs with no neighbours: {len(empty)}")

    def _get_config_exec_time(self, config):
        cache_key = ",".join(str(v) for v in config.values())
        time = self.cache[cache_key]["time"]
        if isinstance(time, str):
            return float("inf")
        return time
