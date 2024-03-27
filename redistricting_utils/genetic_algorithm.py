import itertools
import os

import numpy as np
from shapely.errors import GEOSException

from .algorithm import Algorithm, ParameterCollection, Parameter, RangeParameter
from .maps import count_polygons, save_random_maps, DistrictMap, DistrictMapCollection


def union_from_difference(before, geometries, removals):
    try:
        return before.difference(geometries[removals].unary_union)
    except GEOSException:
        return geometries.unary_union


def apply_bias(a, bias, min_p=0.0):
    """Effective probability balancing algorithm to apply bias exponents to calculate heuristic probabilities."""

    assert -1 <= bias <= 1
    assert 0 <= min_p <= 1

    if len(a) == 0:
        return a
    if len(a) == 1:
        return np.array([1])
    if bias == 0.0:
        return np.full(shape=a.shape, fill_value=1.0)

    epsilon = 1e-10
    exp = (1 / (1 - np.abs(bias) * (1 - epsilon * 2) - epsilon)) - 1

    a_min, a_max = a.min(), a.max()
    if a_max - a_min == 0:
        return np.full(shape=a.shape, fill_value=1.0)
    p = (a - a_min) / (a_max - a_min)

    if bias < 0:
        p = (1 - p)
    p **= exp
    p /= p.sum()

    min_p = min_p / (len(a) - 1)
    p = p * (1 - min_p * len(a)) + min_p

    return p


def calculate_weights(*arrays_and_biases, min_p=0.0):
    """Calculates weights from arrays and biases with the apply_bias algorithm."""
    return np.prod([apply_bias(a, bias, min_p=min_p) for a, bias in arrays_and_biases], axis=0)


def normalize(weights):
    """Normalizes the weights to probabilities."""
    p = np.nan_to_num(weights / weights.sum())
    if p.sum() < 1:
        p[p.argmin()] += 1 - p.sum()
    elif p.sum() > 1:
        p[p.argmax()] -= p.sum() - 1
    return p


class GeneticRedistrictingAlgorithm(Algorithm):
    """Main class the performs the genetic machine learning algorithm to optimize congressional district maps."""

    def __init__(
            self,
            env,
            starting_maps_dir,
            log_path=None,
            verbose=1,
            print_every=1,
            save_every=0,
            population_size=2,
            starting_population_size=None,
            selection_pct=0.5,
            weights=None,
            params=None,
            min_p=0.0,
    ):
        assert verbose >= 0
        assert print_every > 0
        assert save_every >= 0
        assert population_size >= 2
        assert starting_population_size is None or starting_population_size == -1 or \
               starting_population_size >= population_size
        assert selection_pct <= 0.5
        assert 0 <= min_p <= 1

        if params is None:
            params = ParameterCollection()
        params.set_defaults(
            expansion_population_bias=Parameter(0),
            reduction_population_bias=Parameter(0),
            expansion_distance_bias=Parameter(0),
            reduction_distance_bias=Parameter(0),
            expansion_surrounding_bias=Parameter(0),
            reduction_surrounding_bias=Parameter(0),
            mutation_size=RangeParameter(0.0, 1.0),
            mutation_layers=RangeParameter(1, 1),
            mutation_n=RangeParameter(0.0, 1.0),
        )

        super().__init__(env=env, log_path=log_path, verbose=verbose, save_every=save_every, weights=weights,
                         params=params)

        self.print_every = print_every
        self.population_size = population_size
        self.selection_pct = selection_pct
        if starting_population_size is None:
            starting_population_size = population_size
        self.starting_population_size = starting_population_size
        self.starting_maps_dir = starting_maps_dir
        self.min_p = min_p

        self.population = None

    def run(self, generations):
        """Runs the algorithm for a specified number of generations."""
        with self:
            if self._start_map is None:
                self.log(f'Filling population...', verbose=1)
                self.fill_population()
            else:
                self.population.add(self._start_map)

            self.log(f'Simulating for {generations:,} generations...', verbose=1)
            fitness_scores = []
            for generation in range(generations + 1):
                fitness, update = self.simulate_generation(last=generation == generations)
                yield update
                fitness_scores.append(fitness)

            self.log(f'Simulation complete!', verbose=1)

    def fill_population(self):
        """Fills the currently empty population with random maps to begin the algorithm."""
        count = sum(len(os.listdir(os.path.join(self.starting_maps_dir, dir_name)))
                    for dir_name in os.listdir(self.starting_maps_dir))
        start_size = count if self.starting_population_size == -1 else self.starting_population_size
        self.population = DistrictMapCollection(env=self.env, max_size=start_size)
        diff = start_size - count
        self.log(f'Found {count} maps{f", need {diff} more..." if diff > 0 else ""}', verbose=1)
        if diff > 0:
            save_random_maps(env=self.env, save_dir=self.starting_maps_dir, n=diff, start=self.start)
        self.log('Loading maps from saved files...', verbose=1)
        for dir_name in os.listdir(self.starting_maps_dir):
            dir_path = os.path.join(self.starting_maps_dir, dir_name)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                district_map = DistrictMap.load(file_path, env=self.env)
                self.population.add(district_map)
                if self.population.is_full():
                    return

    def simulate_generation(self, last=False):
        """Runs a single generation of the algorithm."""
        should_print = self.time_step_count % self.print_every == 0

        self.log(f'Generation: {self.time_step_count:,} - Calculating fitness scores...',
                 verbose=2 if should_print else None)
        fitness_scores, metrics_list = self.population.calculate_fitness(weights=self.weights)
        self.log(f'Generation: {self.time_step_count:,} - Selecting best individuals...',
                 verbose=2 if should_print else None)
        selected, selected_metrics = self.select(fitness_scores, metrics_list)

        metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in selected_metrics[0].items()]
        self.log(f'Generation: {self.time_step_count:,} - {" | ".join(metric_str for metric_str in metric_strs)}',
                 verbose=1 if should_print else None)

        update = self._tick(selected[0], selected_metrics[0])
        if not last:
            self.log(f'Generation: {self.time_step_count:,} - Mutating for new generation...',
                     verbose=2 if should_print else None)
            self.population = self._mutate(selected)

        return max(fitness_scores.values()), update

    def select(self, fitness_scores, metrics):
        """Selects district maps from the population to mutate based on fitness scores."""
        n = min(max(round(self.population_size * self.selection_pct), 1), self.population_size - 1)
        indices = sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:n]
        selected = self.population.select(indices, new_max_size=self.population_size)
        selected_metrics = [metrics[i] for i in indices]
        return selected, selected_metrics

    def _mutate(self, selected):
        """Mutates the selected district maps. Can perform multiple mutations per map depending on parameters."""
        to_add = DistrictMapCollection(env=self.env, max_size=selected.empty_space())
        for district_map in itertools.cycle(selected):
            district_map = district_map.copy()
            n_mutations = self.params.mutation_n.randint(scale=self.env.n_districts, min_value=1)
            districts = list(range(self.env.n_districts))
            np.random.shuffle(districts)
            for i, district in enumerate(itertools.cycle(districts), 1):
                self._mutation(district_map, district)
                if i == n_mutations:
                    break
            to_add.add(district_map)
            if to_add.is_full():
                break
        selected.add(to_add)
        return selected

    def _mutation(self, district_map, district):
        """Performs a single mutation for a single district map. Will expand or reduce the district depending on what
        is necessary, while being also somewhat random as to which."""
        expand_start, reduce_start = None, None
        for _ in range(self.params.mutation_layers.randint(scale=1, min_value=1)):
            mask = district_map.mask(district)
            centroid = self.env.union_cache.calculate_union(mask).centroid
            population_pct = self.env.data['population'][mask].sum() / self.env.ideal_population
            p = population_pct ** self.params['expansion_population_bias'].value
            p /= (p + 1)
            if np.random.random() < p:
                expand_start = self._expansion(district_map, district, centroid, expand_start)
            else:
                reduce_start = self._reduction(district_map, district, centroid, reduce_start)

    def _expansion(self, district_map, district, centroid, start=None):
        """Performs an expansion mutation for the district."""

        eligible = district_map.get_border(district, reverse=False)
        if eligible.empty:
            return start
        populations = self.env.data['population'].groupby(by=district_map.assignments).sum()[
            district_map.assignments[eligible]].values
        distances = self.env.data.geometry[eligible].distance(centroid).values
        neighbors = self.env.neighbors[district_map.assignments[
            self.env.neighbors['index_right']] == district]['index_right']
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(
            eligible, fill_value=0).values.astype(np.float64)
        weights = calculate_weights(
            (populations, self.params.reduction_population_bias.value),
            (distances, self.params.expansion_distance_bias.value),
            (neighbors, self.params.expansion_surrounding_bias.value),
            min_p=self.min_p,
        )
        selected, start = self._select_mutations(eligible, district_map, weights, centroid, start,
                                                 centroid_distance_weight=1)
        if selected is not None:
            district_map.set(selected, district)
        return start

    def _reduction(self, district_map, district, centroid, start=None):
        """Performs a reduction mutation for the district."""

        eligible = district_map.get_border(district, reverse=True)
        if eligible.empty:
            return start
        populations = self.env.data.geometry[eligible].distance(centroid).values
        neighbors = self.env.neighbors['index_right'][district_map.assignments[
            self.env.neighbors['index_right']] == district]
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(
            eligible, fill_value=0).values.astype(np.float64)
        weights = calculate_weights(
            (populations, self.params.reduction_distance_bias.value),
            (neighbors, self.params.reduction_surrounding_bias.value),
            min_p=self.min_p,
        )
        selected, start = self._select_mutations(eligible, district_map, weights, centroid, start,
                                                 centroid_distance_weight=-1)
        if selected is None:
            return start

        neighbors = self.env.neighbors['index_right'][
            district_map.assignments[self.env.neighbors['index_right']] != district]
        populations = self.env.data['population'].groupby(by=district_map.assignments).sum()
        centroids = {}
        for i in np.unique(district_map.assignments[neighbors]):
            mask = district_map.mask(i)
            geometries = self.env.data.geometry[mask]
            centroids[i] = self.env.union_cache.calculate_union(mask, geometries).centroid

        district_selections, selection = [], None
        for x in selected:
            neighbor_districts = np.unique(district_map.assignments[neighbors[[x]]])
            if selection is None or selection not in neighbor_districts:
                neighbor_centroids = np.array([centroids[i] for i in neighbor_districts])
                populations_ = np.array([populations[i] for i in neighbor_districts])
                distances = self.env.data.geometry[x].distance(neighbor_centroids)
                weights = calculate_weights(
                    (populations_, self.params.expansion_population_bias.value),
                    (distances, self.params.expansion_distance_bias.value),
                    min_p=self.min_p,
                )
                selection = np.random.choice(neighbor_districts, p=normalize(weights))
            district_selections.append(selection)

        district_map.set(selected, district_selections)
        return start

    def _select_mutations(self, eligible, district_map, weights, centroid, start=None, centroid_distance_weight=1):
        """Selects which VTDs to mutate based on parameters while trying to remain contiguous. May return less than
        intended if undesired contiguity harming effects would have been produced with more."""
        if start is None:
            start = self._choose_start(eligible, district_map, weights)
            if start is None:
                return None, None
        eligible = eligible.sort_values(key=lambda x: self.env.data.geometry[x].distance(
            self.env.data.geometry[start]) ** 2 + centroid_distance_weight * self.env.data.geometry[x].distance(
            centroid) ** 2).values[:self.params.mutation_size.randint(len(eligible))]
        selected, bounds = eligible, np.array([0, len(eligible)])
        while True:
            if bounds[0] + 1 >= bounds[1]:
                break
            which_bound = 0
            for i in np.unique(district_map.assignments[selected]):
                removals = selected[district_map.assignments[selected] == i]
                if not self._verify_contiguity(i, district_map, removals):
                    which_bound = 1
                    break
            bounds[which_bound] = len(selected)
            selected = eligible[:int(bounds.mean())]
        return selected, start

    def _choose_start(self, eligible, district_map, weights):
        """Chooses which VTD to start with for a given mutation. Other VTDs are selected by proximity to this starting
        VTD and to the current center of the district."""
        attempts, max_attempts = 0, (weights > 0).sum()
        while attempts < max_attempts:
            i = np.random.choice(range(len(eligible)), p=normalize(weights))
            x = eligible.iloc[i]
            if self._verify_contiguity(district_map.assignments[x], district_map, [x]):
                return x
            weights[i] = 0
            attempts += 1
        return None

    def _verify_contiguity(self, which, district_map, removals):
        """Verifies the contiguity of a district following a potential mutation of VTD assignments to determine whether
        it needs to be modified or rejected entirely."""
        previous = district_map.districts.geometry.loc[which]
        geometries = self.env.data.geometry[district_map.mask(which)]
        new = union_from_difference(previous, geometries, removals)
        a = count_polygons(new) <= count_polygons(previous)
        return a
