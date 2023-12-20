import copy
import datetime as dt
import itertools

import matplotlib.pyplot as plt
import numpy as np
from shapely.errors import GEOSException

from maps import count_polygons, DistrictMapCollection
from redistricting_env import refresh_data, RedistrictingEnv


def time(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


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


class RedistrictingGeneticAlgorithm:
    def __init__(
            self,
            env,
            start=dt.datetime.now(),
            verbose=1,
            save_every=0,
            log_path='log.txt',
            weights=None,
            population_size=100,
            selection_pct=0.5,
            mutation_n_range=(0.0, 1.0),
            mutation_layer_range=(1, 1),
            mutation_size_range=(0.0, 1.00),
            expansion_population_bias=0,
            reduction_population_bias=0,
            expansion_distance_bias=0,
            reduction_distance_bias=0,
            expansion_surrounding_bias=0,
            reduction_surrounding_bias=0,
            starting_population_size=None,
            mutation_n_growth=1.0,
            mutation_size_decay=1.0,
            bias_decay=1.0,
    ):
        if selection_pct > 0.5:
            raise ValueError('Parameter `selection_pct` must be less than or equal to `0.5`.')

        self.env = env

        self.verbose = verbose
        self.start = start
        self.save_every = save_every
        self.log_path = log_path
        open(self.log_path, 'w').close()

        self.weights = {
            'contiguity': 0,
            'population_balance': 1,
            'compactness': 1,
            'win_margin': -1,
            'efficiency_gap': -1,
        }
        for key in weights:
            if key in self.weights:
                self.weights[key] = weights[key]

        self.population_size = population_size
        self.selection_pct = selection_pct
        self.mutation_n_range = mutation_n_range
        self.mutation_layer_range = mutation_layer_range
        self.mutation_size_range = mutation_size_range
        self.expansion_population_bias = expansion_population_bias
        self.reduction_population_bias = reduction_population_bias
        self.expansion_distance_bias = expansion_distance_bias
        self.reduction_distance_bias = reduction_distance_bias
        self.expansion_surrounding_bias = expansion_surrounding_bias
        self.reduction_surrounding_bias = reduction_surrounding_bias
        if starting_population_size is None:
            starting_population_size = population_size
        self.starting_population_size = starting_population_size
        self.mutation_n_growth = mutation_n_growth
        self.mutation_size_decay = mutation_size_decay
        self.bias_decay = bias_decay

        self.population = DistrictMapCollection(env=self.env, size=self.starting_population_size)

        self.generation = 0
        self.current_mutation_n_range = mutation_n_range
        self.current_mutation_size_range = mutation_size_range
        self.current_expansion_population_bias = expansion_population_bias
        self.current_reduction_population_bias = reduction_population_bias
        self.current_expansion_distance_bias = expansion_distance_bias
        self.current_reduction_distance_bias = reduction_distance_bias
        self.current_expansion_surrounding_bias = expansion_surrounding_bias
        self.current_reduction_surrounding_bias = reduction_surrounding_bias

    def run(self, generations=1):
        if self.env.live_plot:
            plt.ion()

        self._log(f'Filling population...')
        self.population.randomize()

        self._log(f'Simulating for {generations:,} generations...')
        for generation in range(generations + 1):
            self.simulate_generation(last=generation == generations)

        self._log(f'Simulation complete!')
        if self.env.live_plot:
            plt.ioff()
            plt.show()

    def _log(self, message):
        message = f'{time(self.start)} - {message}'
        with open(self.log_path, 'a') as f:
            f.write(f'{message}\n')
        if self.verbose:
            print(message)

    def simulate_generation(self, last=False):
        self._log(f'Generation: {self.generation:,} - Calculating fitness scores...')
        fitness_scores, metrics_list = self.calculate_fitness()
        self._log(f'Generation: {self.generation:,} - Selecting best individuals...')
        selected, selected_metrics = self.select(fitness_scores, metrics_list)

        metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in selected_metrics[0].items()]
        self._log(f'Generation: {self.generation:,} - {" | ".join(metric_str for metric_str in metric_strs)}')

        self._log(f'Generation: {self.generation:,} - Plotting best map...')
        self._plot(selected[0])

        if not last:
            self._log(f'Generation: {self.generation:,} - Mutating for new generation...')
            self.population = self._mutate(selected)
            self.generation += 1

            self.current_mutation_n_range = (
                self.current_mutation_n_range[0] * self.mutation_n_growth,
                self.current_mutation_n_range[1] * self.mutation_n_growth,
            )
            self.current_mutation_size_range = (
                self.current_mutation_size_range[0] * self.mutation_size_decay,
                self.current_mutation_size_range[1] * self.mutation_size_decay,
            )
            self.current_expansion_population_bias = self.current_expansion_population_bias * self.bias_decay
            self.current_reduction_population_bias = self.current_reduction_population_bias * self.bias_decay
            self.current_expansion_distance_bias = self.current_expansion_distance_bias * self.bias_decay
            self.current_reduction_distance_bias = self.current_reduction_distance_bias * self.bias_decay
            self.current_expansion_surrounding_bias = self.current_expansion_surrounding_bias * self.bias_decay
            self.current_reduction_surrounding_bias = self.current_reduction_surrounding_bias * self.bias_decay

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
        selected, selected_metrics = [], []
        n = min(max(round(self.population_size * self.selection_pct), 1), self.population_size - 1)
        for i in sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:n]:
            selected.append(self.population[i])
            selected_metrics.append(metrics[i])

        return selected, selected_metrics

    def _plot(self, district_map):
        save = self.env.save_dir is not None and (self.save_every > 0 and self.generation % self.save_every == 0)
        save_path = f'{self.env.save_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}-{self.generation}'
        district_map.plot(save=save, save_path=save_path)

    def _mutate(self, selected):
        population = copy.deepcopy(selected)
        for assignments in itertools.cycle(selected):
            assignments = assignments.copy()
            n_mutations = np.random.randint(
                low=max(int(self.current_mutation_n_range[0] * self.env.n_districts), 1),
                high=max(int(self.current_mutation_n_range[1] * self.env.n_districts), 1) + 1)
            districts = list(range(self.env.n_districts))
            np.random.shuffle(districts)
            for i, x in enumerate(itertools.cycle(districts), 1):
                self._mutation(assignments, x)
                if i == n_mutations:
                    break
            population.append(assignments)
            if len(population) == self.population_size:
                break
        return population

    def _mutation(self, assignments, district_idx):
        expand_start, reduce_start = None, None
        for _ in range(np.random.randint(low=max(self.mutation_layer_range[0], 1),
                                         high=max(self.mutation_layer_range[1], 1) + 1)):
            centroid = self.env.union_cache.calculate_union(assignments == district_idx).centroid
            population_pct = self.env.data['population'][assignments == district_idx].sum() / self.env.ideal_population
            p = calculate_p(np.array([population_pct ** self.current_expansion_population_bias, 1]))
            if np.random.random() < p[0]:
                expand_start = self._expansion(assignments, district_idx, centroid, expand_start)
            else:
                reduce_start = self._reduction(assignments, district_idx, centroid, reduce_start)

    def _expansion(self, assignments, district_idx, centroid, start=None):
        eligible = self._get_border(assignments, district_idx)
        if eligible.empty:
            return start
        weights = (self.env.data['population'].groupby(by=assignments).sum()[
            assignments[eligible]] ** self.current_reduction_population_bias).values
        weights *= (self.env.data.geometry[eligible].distance(centroid) ** self.current_expansion_distance_bias).values
        neighbors = self.env.neighbors[assignments[self.env.neighbors['index_right']] == district_idx]['index_right']
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors ** self.current_expansion_surrounding_bias).values
        selected, start = self._select_mutations(eligible, assignments, calculate_p(weights), centroid, start,
                                                 centroid_distance_weight=1)
        if selected is not None:
            assignments[selected] = district_idx
        return start

    def _reduction(self, assignments, district_idx, centroid, start=None):
        eligible = self._get_border(assignments, district_idx, reverse=True)
        if eligible.empty:
            return start
        weights = (self.env.data.geometry[eligible].distance(centroid) ** self.current_reduction_distance_bias).values
        neighbors = self.env.neighbors['index_right'][assignments[self.env.neighbors['index_right']] == district_idx]
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors.astype(np.float64) ** self.current_reduction_surrounding_bias).values
        selected, start = self._select_mutations(eligible, assignments, calculate_p(weights), centroid, start,
                                                 centroid_distance_weight=-1)
        if selected is None:
            return start

        neighbors = self.env.neighbors['index_right'][assignments[self.env.neighbors['index_right']] != district_idx]
        populations = self.env.data['population'].groupby(by=assignments).sum()
        centroids = {}
        for i in np.unique(assignments[neighbors]):
            mask = assignments == i
            geometries = self.env.data.geometry[mask]
            centroids[i] = self.env.union_cache.calculate_union(mask, geometries).centroid
        district_selections, selection = [], None
        for x in selected:
            neighbor_districts = np.unique(assignments[neighbors[[x]]])
            if selection is None or selection not in neighbor_districts:
                neighbor_centroids = np.array([centroids[i] for i in neighbor_districts])
                weights = np.array([populations[i] for i in neighbor_districts]) ** \
                    self.current_expansion_population_bias * (
                    self.env.data.geometry[x].distance(neighbor_centroids) ** self.current_expansion_distance_bias)
                p = calculate_p(weights)
                selection = np.random.choice(neighbor_districts, p=p)
            district_selections.append(selection)

        assignments[selected] = district_selections
        return start

    def _get_border(self, assignments, which, reverse=False):
        if reverse:
            i1, i2 = self.env.neighbors['index_right'], self.env.neighbors.index
        else:
            i1, i2 = self.env.neighbors.index, self.env.neighbors['index_right']
        return self.env.neighbors[(assignments[i1] == which) &
                                  (assignments[i2] != which)]['index_right'].drop_duplicates()

    def _select_mutations(self, eligible, assignments, p, centroid, start=None, centroid_distance_weight=1):
        previous_unions = {}
        if start is None:
            start = self._choose_start(eligible, assignments, previous_unions, p)
            if start is None:
                return None, None
        eligible = eligible.sort_values(key=lambda x: self.env.data.geometry[x].distance(
            self.env.data.geometry[start]) ** 2 + centroid_distance_weight * self.env.data.geometry[x].distance(
            centroid) ** 2).values[:self._rand_mutation_size(len(eligible))]
        selected, bounds = eligible, np.array([0, len(eligible)])
        while True:
            if bounds[0] + 1 >= bounds[1]:
                break
            which_bound = 0
            for i in np.unique(assignments[selected]):
                if not self._verify_contiguity(previous_unions, i, assignments, selected[assignments[selected] == i]):
                    which_bound = 1
                    break
            bounds[which_bound] = len(selected)
            selected = eligible[:int(bounds.mean())]
        return selected, start

    def _choose_start(self, eligible, assignments, previous_unions, p):
        attempts, max_attempts = 0, (p > 10e-2).sum()
        while attempts < max_attempts:
            i = np.random.choice(range(len(eligible)), p=p)
            x = eligible.iloc[i]
            if self._verify_contiguity(previous_unions, assignments[x], assignments, [x]):
                return x
            p[i] = 0
            p = calculate_p(p)
            attempts += 1
        return None

    def _verify_contiguity(self, previous_unions, which, assignments, removals):
        mask = assignments == which
        geometries = self.env.data.geometry[mask]
        previous = previous_unions.get(which)
        if previous is None:
            previous = previous_unions[which] = self.env.union_cache.calculate_union(mask, geometries)
        new = union_from_difference(previous, geometries, removals)
        return count_polygons(new) <= count_polygons(previous)

    def _rand_mutation_size(self, start_size):
        return np.random.randint(low=max(int(self.current_mutation_size_range[0] * start_size), 1),
                                 high=max(int(self.current_mutation_size_range[1] * start_size), 1) + 1)


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
        weights={
            'contiguity': 0,
            'population_balance': 10,
            'compactness': 2,
            'win_margin': -2,
            'efficiency_gap': -1,
        },
        population_size=2,
        selection_pct=0.5,
        mutation_n_range=(0.0, 1 / 17),
        mutation_layer_range=(1, 10),
        mutation_size_range=(0.0, 1.0),
        expansion_population_bias=-10,
        reduction_population_bias=10,
        expansion_distance_bias=-10,
        reduction_distance_bias=10,
        expansion_surrounding_bias=2,
        reduction_surrounding_bias=-2,
        starting_population_size=1_000,
        mutation_n_growth=1 ** (1 / 20_000),
        mutation_size_decay=1 ** (1 / 10_000),  # TODO: Make decay and growth auto detected as necessary based
        bias_decay=1.0,
    )

    algorithm.run(generations=20_000)


if __name__ == '__main__':
    main()
