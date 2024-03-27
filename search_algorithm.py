import datetime as dt

import numpy as np

from algorithm import Algorithm
from maps import DistrictMap, DISTRICT_FEATURES


class SearchRedistrictingAlgorithm(Algorithm):
    """Simplistic algorithm that uses search techniques to try to optimize the map, built for comparison purposes. Fails
    astoundingly when compared to the more advanced Genetic algorithm, so mostly abandoned from further improvement."""

    def __init__(
            self,
            env,
            verbose=1,
            save_every=1,
            log_path='log.txt',
            weights=None,
    ):
        super().__init__(env=env, verbose=verbose, save_every=save_every, log_path=log_path, weights=weights)

        self.district_map = DistrictMap(env)
        self.metrics = {key: None for key in self.weights}
        self.fitness = 0
        self.mutation_count = 0

    def run(self, generations):
        with self:
            self.log(f'Initiating map...')
            if self._start_map is None:
                self.district_map.randomize()
            else:
                self.district_map = self._start_map
            self._calculate_fitness()

            self.log(f'Simulating for {generations:,} generations...')
            fitness_scores = []
            for generation in range(generations + 1):
                fitness_scores.append(self.fitness)
                self.simulate_generation(last=generation == generations)

            self.log(f'Simulation complete!')
            return fitness_scores

    def _calculate_fitness(self):
        self.fitness = 0
        for metric in self.metrics:
            self.metrics[metric] = getattr(self.district_map, f'calculate_{metric}')()
            self.fitness += self.metrics[metric] * self.weights[metric]

    def simulate_generation(self, last=False):
        metric_strs = [f'{" ".join(key.title().split("_"))}: {value:.4%}' for key, value in self.metrics.items()]
        self.log(f'Generation: {self.time_step_count:,} - Mutations: {self.mutation_count} - '
                 f'Fitness: {self.fitness:.4f} - {" | ".join(metric_str for metric_str in metric_strs)}')

        self._tick(self.district_map)
        if not last:
            self.log(f'Mutating...')
            self._mutate()

    def _mutate(self):
        district_map = self.district_map.copy()

        fitness_change, mutation_count = 0, 0
        while fitness_change <= 0:
            fitness_change = self._mutation(district_map)
            mutation_count += 1
            self.log(f'Mutation Count: {mutation_count} - Proposed Fitness Change: {fitness_change:.4f}')

        self.district_map = district_map
        self._calculate_fitness()
        self.mutation_count += mutation_count

    def _mutation(self, district_map):
        eligible = district_map.get_borders()
        if eligible.empty:
            return None

        from_district = district_map.assignments[eligible]
        to_district = district_map.assignments[eligible.index]
        large_district_map = district_map.repeated(len(eligible))
        range_array = np.array(range(len(eligible))) * self.env.n_districts

        sum_feature_differences = self.env.data[DISTRICT_FEATURES].iloc[eligible].values
        large_district_map.districts.loc[range_array + from_district, DISTRICT_FEATURES] = district_map[
            DISTRICT_FEATURES].iloc[from_district].values - sum_feature_differences
        large_district_map.districts.loc[range_array + to_district, DISTRICT_FEATURES] = district_map[
            DISTRICT_FEATURES].iloc[to_district].values + sum_feature_differences

        geometries = self.env.data.geometry[eligible].reset_index(drop=True)
        large_district_map.districts.geometry.iloc[range_array + from_district] = district_map.geometry.iloc[
            from_district].reset_index(drop=True).difference(geometries).values
        large_district_map.districts.geometry.iloc[range_array + to_district] = district_map.geometry.iloc[
            to_district].reset_index(drop=True).union(geometries).values

        fitness_changes = self._calculate_groupby_fitness(large_district_map) - self.fitness

        i = np.argmax(fitness_changes)
        self.district_map.set(i, to_district[i])

        return fitness_changes[i]

    def _calculate_groupby_fitness(self, district_map):
        outcomes = district_map.calculate_groupby_metrics
        fitness = np.full(len(outcomes[list(outcomes.keys())[0]]), 0)
        for metric, outcome in outcomes.items():
            fitness += outcome * self.weights[metric]
        return fitness

    def calculate_fitness(self, district_map):
        return sum(getattr(district_map, f'calculate_{metric}')() * self.weights[metric] for metric in self.metrics)
