import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection

from maps import DistrictMap, DISTRICT_FEATURES
from redistricting_env import refresh_data, RedistrictingEnv


def time(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


def count_polygons(geometry):
    if isinstance(geometry, (Polygon, LineString)):
        return 1
    elif isinstance(geometry, (MultiPolygon, MultiLineString, GeometryCollection)):
        return len(geometry.geoms)
    else:
        raise ValueError(f'Incorrect geom type: "{type(geometry)}"')


def get_utm_zone(longitude):
    return int(((longitude + 180) / 6) % 60) + 1


def infer_utm_crs(data):
    centroid = data.to_crs(epsg=4326).unary_union.centroid
    zone_number = get_utm_zone(centroid.x)
    hemisphere_prefix = 326 if centroid.y >= 0 else 327
    return f'EPSG:{hemisphere_prefix}{zone_number:02d}'


class RedistrictingSearchAlgorithm(Algorithm):
    def __init__(
            self,
            env,
            start=dt.datetime.now(),
            verbose=1,
            save_every=1,
            log_path='log.txt',
            weights=None,
    ):
        super().__init__(env=env, start=start, verbose=verbose, save_every=save_every, log_path=log_path,
                         weights=weights)

        self.district_map = DistrictMap(env)
        self.metrics = {key: None for key in self.weights}
        self.fitness = 0
        self.mutation_count = 0

    def run(self, generations=1):
        with self:
            self._log(f'Initiating map...')
            self.district_map.randomize()
            self._calculate_fitness()

            self._log(f'Simulating for {generations:,} generations...')
            for generation in range(generations + 1):
                self._simulate_generation(last=generation == generations)

            self._log(f'Simulation complete!')

    def _log(self, message):
        message = f'{time(self.start)} - {message}'
        with open(self.log_path, 'a') as f:
            f.write(f'{message}\n')
        if self.verbose:
            print(message)

    def _calculate_fitness(self):
        self.fitness = 0
        for metric in self.metrics:
            self.metrics[metric] = getattr(self.district_map, f'calculate_{metric}')()
            self.fitness += self.metrics[metric] * self.weights[metric]

    def _simulate_generation(self, last=False):
        metric_strs = [f'{" ".join(key.title().split("_"))}: {value:.4%}' for key, value in self.metrics.items()]
        self._log(f'Generation: {self.generation_count:,} - Mutations: {self.mutation_count} - '
                  f'Fitness: {self.fitness:.4f} - {" | ".join(metric_str for metric_str in metric_strs)}')

        self._tick(self.district_map)
        if not last:
            self._log(f'Mutating...')
            self.mutate()

    def mutate(self):
        district_map = self.district_map.copy()

        fitness_change, mutation_count = 0, 0
        while fitness_change <= 0:
            fitness_change = self._mutation(district_map)
            mutation_count += 1
            self._log(f'Mutation Count: {mutation_count} - Proposed Fitness Change: {fitness_change:.4f}')

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


def main():
    start = dt.datetime.now()

    refresh = False
    if refresh:
        print(f'{time(start)} - Refreshing data...')
        refresh_data()

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingSearchAlgorithm(
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
    )

    algorithm.run(generations=20_000)


if __name__ == '__main__':
    main()
