import copy
import datetime as dt

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.ops
import topojson as tp
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection


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


class RedistrictingSearchAlgorithm:
    def __init__(
            self,
            data_path,
            start=dt.datetime.now(),
            verbose=1,
            save_dir=None,
            save_every=1,
            log_path='log.txt',
            live_plot=True,
            n_districts=17,
            weights=None,
    ):
        self.data = gpd.read_parquet(data_path)
        numeric_cols = self.data.select_dtypes(np.number).columns
        self.data[numeric_cols] = self.data[numeric_cols].astype(np.float64)

        self.neighbors = gpd.sjoin(self.data, self.data, how='inner', predicate='touches')
        self.neighbors = self.neighbors[self.neighbors.index != self.neighbors['index_right']]

        self.verbose = verbose
        self.start = start
        self.save_dir = save_dir
        self.save_every = save_every
        self.log_path = log_path
        open(self.log_path, 'w').close()
        self.live_plot = live_plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.n_districts = n_districts
        self.ideal_pop = self.data['population'].sum() / self.n_districts

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

        self.assignments = np.full(len(self.data), -1)
        self.sum_features = pd.Index(['population', 'democrat', 'republican'])
        self.district_map = gpd.GeoDataFrame(
            data=np.full((n_districts, 3), np.nan),
            columns=self.sum_features,
            geometry=np.full(n_districts, Polygon(np.zeros(6).reshape(3, 2))),
            crs=self.data.crs,
        )
        self.metrics = {key: None for key in self.weights}
        self.fitness = 0
        self.mutation_count = 0
        self.generation_count = 0

    def run(self, generations=1):
        if self.live_plot:
            plt.ion()

        self._log(f'Initiating map...')
        self._initiate_map()
        self._calculate_fitness()

        self._log(f'Simulating for {generations:,} generations...')
        for generation in range(generations + 1):
            self._simulate_generation(last=generation == generations)

        self._log(f'Simulation complete!')
        if self.live_plot:
            plt.ioff()
            plt.show()

    def _log(self, message):
        message = f'{self._time()} - {message}'
        with open(self.log_path, 'a') as f:
            f.write(f'{message}\n')
        if self.verbose:
            print(message)

    def _time(self):
        return f'{dt.timedelta(seconds=round((dt.datetime.now() - self.start).total_seconds()))}'

    def _initiate_map(self):
        n_allocated = 0
        for district, starting_point in enumerate(np.random.choice(len(self.data), self.n_districts, replace=False)):
            self.assignments[starting_point] = district
            n_allocated += 1

        skipped_districts, n_allocated = set(), self.n_districts
        districts, neighbors = np.array(range(self.n_districts)), self.neighbors['index_right']
        while n_allocated < len(self.data) and len(skipped_districts) != self.n_districts:
            np.random.shuffle(districts)
            for district in districts:
                if district in skipped_districts:
                    continue

                touching_vtds = neighbors[(self.assignments[neighbors.index] == district) &
                                          (self.assignments[neighbors] == -1)].unique()
                if not len(touching_vtds):
                    skipped_districts.add(district)
                    continue

                self.assignments[touching_vtds] = district
                n_allocated += len(touching_vtds)

        while n_allocated < len(self.data):
            for i in np.where(self.assignments == -1)[0]:
                unallocated = self.data.geometry[i]
                closest_vtd = np.argmin(unallocated.distance(self.data.geometry[self.assignments != -1]))
                if self.assignments[closest_vtd] == -1:
                    continue
                self.assignments[i] = self.assignments[closest_vtd]
                n_allocated += 1

        districts = self.data.groupby(by=self.assignments)
        self.district_map[self.sum_features] = districts[self.sum_features].sum()
        self.district_map.geometry = districts.geometry.apply(shapely.ops.unary_union)

    def _calculate_fitness(self):
        self.fitness = 0
        for metric in self.metrics:
            self.metrics[metric] = getattr(self, f'calculate_{metric}')(self.district_map)
            self.fitness += self.metrics[metric] * self.weights[metric]

    def calculate_contiguity(self, district_map):
        return 1 - sum(district_map.geometry.apply(count_polygons) - 1) / (
            len(self.data) - self.n_districts)

    def calculate_population_balance(self, district_map):
        return (np.minimum(district_map['population'], self.ideal_pop) /
                np.maximum(district_map['population'], self.ideal_pop)).mean()

    @staticmethod
    def calculate_compactness(district_map):
        return (4 * np.pi * district_map.geometry.area / district_map.geometry.length ** 2).mean()

    @staticmethod
    def calculate_win_margin(district_map):
        return ((district_map['democrat'] - district_map['republican']).abs() /
                (district_map['democrat'] + district_map['republican'])).mean()

    @staticmethod
    def calculate_efficiency_gap(district_map):
        total_votes = district_map['democrat'] + district_map['republican']
        votes_needed = np.floor(total_votes) + 1
        rep_wv = (district_map['republican'] - (votes_needed * (
            district_map['republican'] >= district_map['democrat']))).sum()
        dem_wv = (district_map['democrat'] - (votes_needed * (
            district_map['democrat'] >= district_map['republican']))).sum()
        return np.abs(dem_wv - rep_wv) / total_votes.sum()

    def _simulate_generation(self, last=False):
        metric_strs = [f'{" ".join(key.title().split("_"))}: {value:.4%}' for key, value in self.metrics.items()]
        self._log(f'Generation: {self.generation_count:,} - Mutations: {self.mutation_count} - '
                  f'Fitness: {self.fitness:.4f} - {" | ".join(metric_str for metric_str in metric_strs)}')

        self._log(f'Plotting...')
        self._plot()

        if not last:
            self._log(f'Mutating...')
            self.mutate()

    def mutate(self):
        assignments = self.assignments.copy()
        district_map = copy.deepcopy(self.district_map)

        fitness_change, mutation_count = 0, 0
        while fitness_change <= 0:
            fitness_change = self._mutation(assignments, district_map)
            mutation_count += 1

        self.assignments = assignments
        self.district_map = district_map
        self._calculate_fitness()
        self.mutation_count += mutation_count
        self.generation_count += 1

    def _mutation(self, assignments, district_map):
        eligible = self._get_border(assignments)
        if eligible.empty:
            return None

        from_district = assignments[eligible]
        to_district = assignments[eligible.index]
        large_district_map = pd.concat([district_map for _ in range(len(eligible))], ignore_index=True)
        range_array = np.array(range(len(eligible))) * self.n_districts

        sum_feature_differences = self.data[self.sum_features].iloc[eligible].values
        large_district_map.loc[range_array + from_district, self.sum_features] = district_map[
            self.sum_features].iloc[from_district].values - sum_feature_differences
        large_district_map.loc[range_array + to_district, self.sum_features] = district_map[
            self.sum_features].iloc[to_district].values + sum_feature_differences

        geometries = self.data.geometry[eligible].reset_index()
        large_district_map.geometry.iloc[range_array + from_district] = district_map.geometry.iloc[
            from_district].reset_index().difference(geometries).values
        large_district_map.geometry.iloc[range_array + to_district] = district_map.geometry.iloc[
            to_district].reset_index().union(geometries).values

        # TODO: Ensure changes don't affect contiguity
        fitness_changes = self._calculate_groupby_fitness(large_district_map) - self.fitness

        i = np.argmax(fitness_changes)
        from_i, to_i = from_district[i], to_district[i]
        index = district_map.index[np.array([from_i, to_i])]
        assignments[eligible.iloc[i]] = to_i
        district_map.loc[index, self.sum_features] = large_district_map[
            self.sum_features].iloc[np.array([from_i, to_i]) + i * self.n_districts].values
        district_map.geometry.loc[index] = large_district_map.geometry[
            np.array([from_i, to_i]) + i * self.n_districts].values

        return fitness_changes[i]

    def _get_border(self, assignments):
        return self.neighbors[(assignments[self.neighbors.index] != assignments[
            self.neighbors['index_right']])]['index_right']

    def _calculate_groupby_fitness(self, district_map):
        for metric in self.metrics:
            getattr(self, f'_calculate_groupby_{metric}')(district_map)
        group_by = district_map.groupby(by=district_map.index // self.n_districts)
        fitness = np.full(len(group_by), 0)
        for metric in self.metrics:
            if metric == 'efficiency_gap':
                outcome = np.abs(group_by['dem_wv'].sum() - group_by['rep_wv'].sum()) / group_by['total_votes'].sum()
            else:
                outcome = group_by[metric].mean()
            fitness += outcome * self.weights[metric]
        return fitness

    def _calculate_groupby_contiguity(self, district_map):
        district_map['contiguity'] = 1 - sum(district_map.geometry.apply(count_polygons) - 1) / (
            len(self.data) - self.n_districts)

    def _calculate_groupby_population_balance(self, district_map):
        district_map['population_balance'] = np.minimum(district_map['population'], self.ideal_pop) / np.maximum(
            district_map['population'], self.ideal_pop)

    @staticmethod
    def _calculate_groupby_compactness(district_map):
        district_map['compactness'] = 4 * np.pi * district_map.geometry.area / district_map.geometry.length ** 2

    @staticmethod
    def _calculate_groupby_win_margin(district_map):
        district_map['win_margin'] = (district_map['democrat'] - district_map['republican']).abs() / (
            district_map['democrat'] + district_map['republican'])

    @staticmethod
    def _calculate_groupby_efficiency_gap(district_map):
        district_map['total_votes'] = district_map['democrat'] + district_map['republican']
        votes_needed = np.floor(district_map['total_votes']) + 1
        district_map['rep_wv'] = district_map['republican'] - (
            votes_needed * (district_map['republican'] >= district_map['democrat']))
        district_map['dem_wv'] = district_map['democrat'] - (
            votes_needed * (district_map['democrat'] >= district_map['republican']))

    def calculate_fitness(self, district_map):
        return sum(getattr(self, f'calculate_{metric}')(district_map) * self.weights[metric] for metric in self.metrics)

    def _plot(self):
        save = self.save_dir is not None and self.save_every > 0 and self.generation_count % self.save_every == 0
        if not (save or self.live_plot):
            return

        self.ax.clear()
        self.data['_temp_district'] = self.assignments
        self.data.plot(column='_temp_district', ax=self.ax, cmap='tab20')

        if self.live_plot:
            plt.draw()
            plt.pause(0.10)

        if save:
            self.fig.savefig(f'{self.save_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}-{self.generation_count}')


def main():
    start = dt.datetime.now()

    refresh_data = False
    if refresh_data:
        print(f'{time(start)} - Refreshing data...')
        data = gpd.read_file('data/pa/vtd-election-and-census-data-14-20.shp')
        data.to_crs(infer_utm_crs(data), inplace=True)
        data.rename(columns={'TOTPOP20': 'population', 'PRES16D': 'democrat', 'PRES16R': 'republican'}, inplace=True)
        data = tp.Topology(data, prequantize=False).toposimplify(50).to_gdf()
        data.geometry[~data.geometry.is_valid] = data.geometry[~data.geometry.is_valid].buffer(0)
        data.to_parquet('data/pa/simplified.parquet')

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingSearchAlgorithm(
        data_path='data/pa/simplified.parquet',
        start=start,
        verbose=True,
        save_dir='maps',
        save_every=100,
        log_path='log.txt',
        live_plot=False,
        n_districts=17,
        weights={
            'contiguity': 0,
            'population_balance': 4,
            'compactness': 1,
            'win_margin': -1,
            'efficiency_gap': -1,
        },
    )

    algorithm.run(generations=20_000)


if __name__ == '__main__':
    main()
