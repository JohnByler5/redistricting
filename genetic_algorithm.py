import datetime as dt
import itertools

import numpy as np
from shapely.errors import GEOSException

from algorithm import time, Algorithm, DictCollection, ParameterCollection, Parameter, RangeParameter
from maps import count_polygons, DistrictMapCollection
from redistricting_env import refresh_data, RedistrictingEnv


def union_from_difference(before, geometries, removals):
    try:
        return before.difference(geometries[removals].unary_union)
    except GEOSException:
        return geometries.unary_union


def calculate_p(weights):
    p = np.nan_to_num(weights / weights.sum())
    if p.sum() != 1:
        p[np.argmax(p)] += 1 - p.sum()
    return p


class RedistrictingGeneticAlgorithm(Algorithm):
    def __init__(
            self,
            env,
            start=dt.datetime.now(),
            verbose=1,
            save_every=0,
            log_path='log.txt',
            weights=None,
            params=None,
            population_size=2,
            selection_pct=0.5,
            starting_population_size=None,
    ):
        assert selection_pct <= 0.5

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

        super().__init__(env=env, start=start, log_path=log_path, verbose=verbose, save_every=save_every,
                         weights=weights, params=params)

        self.population_size = population_size
        self.selection_pct = selection_pct
        if starting_population_size is None:
            starting_population_size = population_size
        self.starting_population_size = starting_population_size

        self.population = DistrictMapCollection(env=self.env, max_size=self.starting_population_size)

    def run(self, generations=1):
        with self:
            self._log(f'Filling population...')
            self.population.randomize()

            self._log(f'Simulating for {generations:,} generations...')
            for generation in range(generations + 1):
                self.simulate_generation(last=generation == generations)

            self._log(f'Simulation complete!')

    def simulate_generation(self, last=False):
        self._log(f'Generation: {self.time_step_count:,} - Calculating fitness scores...')
        fitness_scores, metrics_list = self.calculate_fitness()
        self._log(f'Generation: {self.time_step_count:,} - Selecting best individuals...')
        selected, selected_metrics = self.select(fitness_scores, metrics_list)

        metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in selected_metrics[0].items()]
        self._log(f'Generation: {self.time_step_count:,} - {" | ".join(metric_str for metric_str in metric_strs)}')

        self._tick(selected[0])
        if not last:
            self._log(f'Generation: {self.time_step_count:,} - Mutating for new generation...')
            self.population = self._mutate(selected)

    def calculate_fitness(self):
        scores, metrics = {}, {}
        for i, district_map in enumerate(self.population):
            scores[i] = 0
            metrics[i] = {'fitness': '0'}
            for metric, weight in self.weights.items():
                result = getattr(district_map, f'calculate_{metric}')
                scores[i] += result * weight
                metrics[i][metric] = f'{result:.4%}'
            metrics[i]['fitness'] = f'{scores[i]:.4f}'

        return scores, metrics

    def select(self, fitness_scores, metrics):
        n = min(max(round(self.population_size * self.selection_pct), 1), self.population_size - 1)
        indices = sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:n]
        selected = self.population.select(indices)
        selected_metrics = [metrics[i] for i in indices]
        return selected, selected_metrics

    def _mutate(self, selected):
        for district_map in itertools.cycle(selected):
            district_map = district_map.copy()
            n_mutations = self.params.mutation_n.randint(sacle=self.env.n_districts, min_value=1)
            districts = list(range(self.env.n_districts))
            np.random.shuffle(districts)
            for i, district in enumerate(itertools.cycle(districts), 1):
                self._mutation(district_map, district)
                if i == n_mutations:
                    break
            selected.add(district_map)
            if selected.size == selected.max_size:
                break
        return selected

    def _mutation(self, district_map, district):
        expand_start, reduce_start = None, None
        for _ in range(self.params['mutation_layers'].randint(scale=1, min_value=1)):
            mask = district_map.mask(district)
            centroid = self.env.union_cache.calculate_union(mask).centroid
            population_pct = self.env.data['population'][mask].sum() / self.env.ideal_population
            p = calculate_p(np.array([population_pct ** self.params['expansion_population_bias'].value, 1]))
            if np.random.random() < p[0]:
                expand_start = self._expansion(district_map, district, centroid, expand_start)
            else:
                reduce_start = self._reduction(district_map, district, centroid, reduce_start)

    def _expansion(self, district_map, district, centroid, start=None):
        eligible = district_map.get_border(district, reverse=False)
        if eligible.empty:
            return start
        weights = (self.env.data['population'].groupby(by=district_map.assignments).sum()[
            district_map.assignments[eligible]] ** self.params.reduction_population_bias.value).values
        weights *= (self.env.data.geometry[eligible].distance(centroid) **
                    self.params.expansion_distance_bias.value).values
        neighbors = self.env.neighbors[district_map.assignments[
            self.env.neighbors['index_right']] == district]['index_right']
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors ** self.params.expansion_surrounding_bias.value).values
        selected, start = self._select_mutations(eligible, district_map, calculate_p(weights), centroid, start,
                                                 centroid_distance_weight=1)
        if selected is not None:
            district_map.set(selected, district)
        return start

    def _reduction(self, district_map, district, centroid, start=None):
        eligible = district_map.get_border(district, reverse=False)
        if eligible.empty:
            return start
        weights = (self.env.data.geometry[eligible].distance(centroid) **
                   self.params.reduction_distance_bias.value).values
        neighbors = self.env.neighbors['index_right'][district_map.assignments[
            self.env.neighbors['index_right']] == district]
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors.astype(np.float64) ** self.params.reduction_surrounding_bias.value).values
        selected, start = self._select_mutations(eligible, district_map, calculate_p(weights), centroid, start,
                                                 centroid_distance_weight=-1)
        if selected is None:
            return start

        neighbors = self.env.neighbors['index_right'][
            district_map.assignments[self.env.neighbors['index_right']] != district]
        populations = self.env.data['population'].groupby(by=district_map.assignments).sum()
        centroids = {}
        for i in np.unique(district_map.assignments[neighbors]):
            mask = district_map.get_mask(i)
            geometries = self.env.data.geometry[mask]
            centroids[i] = self.env.union_cache.calculate_union(mask, geometries).centroid
        district_selections, selection = [], None
        for x in selected:
            neighbor_districts = np.unique(district_map.assignments[neighbors[[x]]])
            if selection is None or selection not in neighbor_districts:
                neighbor_centroids = np.array([centroids[i] for i in neighbor_districts])
                weights = np.array([populations[i] for i in neighbor_districts]) ** \
                    self.params.expansion_population_bias.value * (
                    self.env.data.geometry[x].distance(neighbor_centroids) ** self.params.expansion_distance_bias.value)
                p = calculate_p(weights)
                selection = np.random.choice(neighbor_districts, p=p)
            district_selections.append(selection)

        district_map.set(selected, district_selections)
        return start

    def _select_mutations(self, eligible, district_map, p, centroid, start=None, centroid_distance_weight=1):
        previous_unions = {}
        if start is None:
            start = self._choose_start(eligible, district_map, previous_unions, p)
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
                if not self._verify_contiguity(previous_unions, i, district_map.assignments, removals):
                    which_bound = 1
                    break
            bounds[which_bound] = len(selected)
            selected = eligible[:int(bounds.mean())]
        return selected, start

    def _choose_start(self, eligible, district_map, previous_unions, p):
        attempts, max_attempts = 0, (p > 10e-2).sum()
        while attempts < max_attempts:
            i = np.random.choice(range(len(eligible)), p=p)
            x = eligible.iloc[i]
            if self._verify_contiguity(previous_unions, district_map.assignments[x], district_map, [x]):
                return x
            p[i] = 0
            p = calculate_p(p)
            attempts += 1
        return None

    def _verify_contiguity(self, previous_unions, which, district_map, removals):
        mask = district_map.mask(which)
        geometries = self.env.data.geometry[mask]
        previous = previous_unions.get(which)
        if previous is None:
            previous = previous_unions[which] = self.env.union_cache.calculate_union(mask, geometries)
        new = union_from_difference(previous, geometries, removals)
        return count_polygons(new) <= count_polygons(previous)


def main():
    start = dt.datetime.now()

    refresh = False
    if refresh:
        print(f'{time(start)} - Refreshing data...')
        refresh_data()

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingGeneticAlgorithm(
        env=RedistrictingEnv('data/pa/simplified.parquet', n_districts=17, live_plot=False, save_dir='maps'),
        start=start,
        verbose=True,
        save_every=1_000,
        log_path='log.txt',
        weights=DictCollection(
            contiguity=0,
            population_balance=10,
            compactness=2,
            win_margin=-2,
            efficiency_gap=-1,
        ),
        params=ParameterCollection(
            expansion_population_bias=Parameter(-10, exp_factor=1),
            reduction_population_bias=Parameter(10, exp_factor=1),
            expansion_distance_bias=Parameter(-10, exp_factor=1),
            reduction_distance_bias=Parameter(10, exp_factor=1),
            expansion_surrounding_bias=Parameter(2, exp_factor=1),
            reduction_surrounding_bias=Parameter(-2, exp_factor=1),
            mutation_size=RangeParameter(1.0, 1.0, exp_factor=0.01 ** (1 / 20_000)),
            mutation_layers=RangeParameter(20, 20, exp_factor=0.1 ** (1 / 20_000), min_value=1),
            mutation_n=RangeParameter(0.0, 1 / 17, exp_factor=34 ** (1 / 20_000), max_value=34),
        ),
        population_size=2,
        selection_pct=0.5,
        starting_population_size=2,
    )

    algorithm.run(generations=20_000)


if __name__ == '__main__':
    main()
