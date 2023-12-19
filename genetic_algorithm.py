import copy
import datetime as dt
import itertools

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import topojson as tp
from shapely.errors import GEOSException
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection


def time(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


def count_polygons(geometry):
    if isinstance(geometry, Polygon):
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


def count_points_in_geom(geom):
    if isinstance(geom, Polygon):
        return len(geom.exterior.coords) + sum(len(interior.coords) for interior in geom.interiors)
    elif isinstance(geom, LineString):
        return len(geom.coords)
    elif isinstance(geom, (MultiPolygon, MultiLineString, GeometryCollection)):
        return sum(count_points_in_geom(part) for part in geom.geoms)
    else:
        raise AssertionError(f'Incorrect geom type: {type(geom)}')


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
            data_path,
            start=dt.datetime.now(),
            verbose=1,
            save_dir=None,
            save_every=1,
            log_path='log.txt',
            live_plot=True,
            n_districts=17,
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

        self.data = gpd.read_parquet(data_path)
        num_cols = self.data.select_dtypes(np.number).columns
        self.data[num_cols] = self.data[num_cols].astype(np.float64)

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
        self.total_pop = self.data['population'].sum()
        self.ideal_pop = self.total_pop / self.n_districts

        self.weights = {
            'pop_balance': 1,
            'compactness': 1,
            'competitiveness': 1,
            'efficiency_gap': 1,
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

        self.generation = 0
        self.population = []
        self.current_mutation_n_range = mutation_n_range
        self.current_mutation_size_range = mutation_size_range
        self.current_expansion_population_bias = expansion_population_bias
        self.current_reduction_population_bias = reduction_population_bias
        self.current_expansion_distance_bias = expansion_distance_bias
        self.current_reduction_distance_bias = reduction_distance_bias
        self.current_expansion_surrounding_bias = expansion_surrounding_bias
        self.current_reduction_surrounding_bias = reduction_surrounding_bias

        self.union_cache = gpd.GeoDataFrame(data=[[[None for _ in range(len(self.data))]] for _ in range(1_000)],
                                            columns=['mask'], geometry=[None for _ in range(1_000)])
        self.union_cache_count = 0

    def append_union_cache(self, mask, union):
        self.union_cache.at[self.union_cache_count, 'mask'] = mask
        self.union_cache.at[self.union_cache_count, 'geometry'] = union
        self.union_cache_count += 1

        if self.union_cache_count == len(self.union_cache):
            original = self.union_cache[len(self.union_cache) // 2:]
            empty_len = len(self.union_cache) - len(original)
            empty = gpd.GeoDataFrame(data=[[[None for _ in range(len(self.data))]] for _ in range(empty_len)],
                                     columns=['mask'], geometry=[None for _ in range(empty_len)])
            self.union_cache = pd.concat([original, empty]).reset_index(drop=True)
            self.union_cache_count = len(original)

    def fill_population(self, size=None, clear=False):
        if size is None:
            size = self.starting_population_size
        if clear:
            self.population.clear()
        for _ in range(size - len(self.population)):
            self.population.append(self.random_map())

    def random_map(self):
        assignments = np.full(len(self.data), -1)

        n_allocated = 0
        for district, starting_point in enumerate(np.random.choice(len(self.data), self.n_districts, replace=False)):
            assignments[starting_point] = district
            n_allocated += 1

        skipped_districts, n_allocated = set(), self.n_districts
        districts, neighbors = np.array(range(self.n_districts)), self.neighbors['index_right']
        while n_allocated < len(self.data) and len(skipped_districts) != self.n_districts:
            np.random.shuffle(districts)
            for district in districts:
                if district in skipped_districts:
                    continue

                touching_vtds = neighbors[(assignments[neighbors.index] == district) &
                                          (assignments[neighbors] == -1)].unique()
                if not len(touching_vtds):
                    skipped_districts.add(district)
                    continue

                assignments[touching_vtds] = district
                n_allocated += len(touching_vtds)

        while n_allocated < len(self.data):
            for i in np.where(assignments == -1)[0]:
                unallocated = self.data.geometry[i]
                closest_vtd = np.argmin(unallocated.distance(self.data.geometry[assignments != -1]))
                if assignments[closest_vtd] == -1:
                    continue
                assignments[i] = assignments[closest_vtd]
                n_allocated += 1

        return assignments

    def calculate_contiguity(self, district_map):
        total_breaks = sum(district_map['geometry'].apply(count_polygons) - 1)
        max_breaks = len(self.data) - self.n_districts
        return 1 - total_breaks / max_breaks

    def calculate_population_balance(self, district_map):
        return (np.minimum(district_map['population'], self.ideal_pop) / np.maximum(
            district_map['population'], self.ideal_pop)).mean()

    @staticmethod
    def calculate_compactness(district_map):
        return (4 * np.pi * district_map.geometry.area / district_map.geometry.length ** 2).mean()

    @staticmethod
    def calculate_competitiveness(district_map):
        return (np.maximum(district_map['democrat'], district_map['republican']) / (
                district_map['democrat'] + district_map['republican'])).mean()

    @staticmethod
    def calculate_efficiency_gap(district_map):
        district_map['total_votes'] = district_map['democrat'] + district_map['republican']
        district_map['votes_needed'] = np.floor(district_map['total_votes']) + 1
        rep_wv = (district_map['republican'] - (district_map['votes_needed'] * (
            district_map['republican'] >= district_map['democrat']))).sum()
        dem_wv = (district_map['democrat'] - (district_map['votes_needed'] * (
            district_map['democrat'] >= district_map['republican']))).sum()
        return np.abs(dem_wv - rep_wv) / district_map['total_votes'].sum()

    def calculate_bbox_score(self, bounds):
        bounds_series = self.union_cache.geometry[:self.union_cache_count].bounds
        u_minx, u_miny, u_maxx, u_maxy = [bounds_series[s] for s in ['minx', 'miny', 'maxx', 'maxy']]

        intersection_width = np.maximum(np.minimum(u_maxx, bounds[2]) - np.maximum(u_minx, bounds[0]), 0)
        intersection_height = np.maximum(np.minimum(u_maxy, bounds[3]) - np.maximum(u_miny, bounds[1]), 0)
        intersection_area = intersection_width * intersection_height

        u_area = (u_maxx - u_minx) * (u_maxy - u_miny)
        b_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if b_area == 0:
            return np.zeroes(self.union_cache_count)

        scores = (intersection_area * 2 - u_area) / b_area
        scores[u_area == 0] = 0
        return scores.values

    def _find_closest_union(self, bounds):
        scores = self.calculate_bbox_score(bounds)
        if not len(scores):
            return None, None, 0
        i = scores.argmax()
        return self.union_cache.geometry[i], self.union_cache['mask'][i], scores[i]

    def _calculate_union(self, mask, geometries=None):
        if geometries is None:
            geometries = self.data.geometry[mask]
        prev_union, prev_mask, score = self._find_closest_union(geometries.total_bounds)
        if score > 0.2:
            try:
                geometry = prev_union
                to_subtract = self.data.geometry[prev_mask & (~mask)]
                if not to_subtract.empty:
                    geometry = geometry.difference(to_subtract.unary_union)
                to_add = self.data.geometry[mask & (~prev_mask)]
                if not to_add.empty:
                    geometry = geometry.union(to_add.unary_union)
                return geometry
            except GEOSException:
                pass
        return geometries.geometry.unary_union

    def construct_map(self, assignments):
        district_data = []
        for district_index in range(self.n_districts):
            mask = assignments == district_index
            if not any(mask):
                geometry = Polygon(np.zeros(6).reshape(3, 2))
            else:
                geometry = self._calculate_union(mask)
                self.append_union_cache(mask, geometry)

            district_data.append({
                'district': district_index,
                'geometry': geometry,
                'population': self.data['population'][mask].sum(),
                'democrat': self.data['democrat'][mask].sum(),
                'republican': self.data['republican'][mask].sum(),
            })

        return gpd.GeoDataFrame(district_data, crs=self.data.crs)

    def calculate_fitness(self):
        scores, metrics_list = {}, []
        for i, assignments in enumerate(self.population):
            district_map = self.construct_map(assignments)

            contiguity = self.calculate_contiguity(district_map)
            pop_balance = self.calculate_population_balance(district_map)
            compactness = self.calculate_compactness(district_map)
            competitiveness = self.calculate_competitiveness(district_map)
            efficiency_gap = self.calculate_efficiency_gap(district_map)

            scores[i] = pop_balance * self.weights['pop_balance'] + compactness * self.weights['compactness'] + \
                (1 - competitiveness) * 2 * self.weights['competitiveness'] + (1 - efficiency_gap) * \
                self.weights['efficiency_gap']

            metrics_list.append({
                'Fitness': f'{scores[i]:.4f}',
                'Contiguity': f'{contiguity:.4%}',
                'Population Balance': f'{pop_balance:.4%}',
                'Compactness': f'{compactness:.4%}',
                'Competitiveness': f'{competitiveness:.4%}',
                'Efficiency Gap': f'{efficiency_gap:.4%}'
            })

        return scores, metrics_list

    def select(self, fitness_scores, metrics_list):
        selected, selected_metrics = [], []
        n = min(max(round(self.population_size * self.selection_pct), 1), self.population_size - 1)
        for i in sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:n]:
            selected.append(self.population[i])
            selected_metrics.append(metrics_list[i])

        return selected, selected_metrics

    def _rand_mutation_size(self, start_size):
        return np.random.randint(low=max(int(self.current_mutation_size_range[0] * start_size), 1),
                                 high=max(int(self.current_mutation_size_range[1] * start_size), 1) + 1)

    def _check_contiguity(self, previous_unions, which, assignments, removals):
        mask = assignments == which
        geometries = self.data.geometry[mask]
        previous = previous_unions.get(which)
        if previous is None:
            previous = previous_unions[which] = self._calculate_union(mask, geometries)
        new = union_from_difference(previous, geometries, removals)
        return count_polygons(new) <= count_polygons(previous)

    def _choose_start(self, eligible, assignments, previous_unions, p):
        attempts, max_attempts = 0, (p > 10e-3).sum()
        while attempts < max_attempts:
            i = np.random.choice(range(len(eligible)), p=p)
            x = eligible.iloc[i]
            if self._check_contiguity(previous_unions, assignments[x], assignments, [x]):
                return x
            p[i] = 0
            p = calculate_p(p)
            attempts += 1
        return None

    def _select_mutations(self, eligible, assignments, p, centroid, start=None, centroid_distance_weight=1):
        previous_unions = {}
        if start is None:
            start = self._choose_start(eligible, assignments, previous_unions, p)
            if start is None:
                return None, None
        eligible = eligible.sort_values(key=lambda x: self.data.geometry[x].distance(
            self.data.geometry[start]) ** 2 + centroid_distance_weight * self.data.geometry[x].distance(
            centroid) ** 2).values[:self._rand_mutation_size(len(eligible))]
        selected, bounds = eligible.copy(), np.array([0, len(eligible)])
        while True:
            if bounds[0] + 1 >= bounds[1]:
                break
            which_bound = 0
            for i in np.unique(assignments[selected]):
                if not self._check_contiguity(previous_unions, i, assignments, selected[assignments[selected] == i]):
                    which_bound = 1
                    break
            bounds[which_bound] = len(selected)
            selected = eligible[:int(bounds.mean())]
        return selected, start

    def _get_border(self, assignments, which, reverse=False):
        if reverse:
            i1, i2 = self.neighbors['index_right'], self.neighbors.index
        else:
            i1, i2 = self.neighbors.index, self.neighbors['index_right']
        return self.neighbors[(assignments[i1] == which) & (assignments[i2] != which)]['index_right'].drop_duplicates()

    def _expansion(self, assignments, district_idx, centroid, start=None):
        eligible = self._get_border(assignments, district_idx)
        if eligible.empty:
            return start
        weights = (self.data['population'].groupby(by=assignments).sum()[
            assignments[eligible]] ** self.current_reduction_population_bias).values
        weights *= (self.data.geometry[eligible].distance(self.data.geometry[
            assignments == district_idx].unary_union.centroid) ** self.current_expansion_distance_bias).values
        neighbors = self.neighbors[assignments[self.neighbors['index_right']] == district_idx]['index_right']
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
        weights = (self.data.geometry[eligible].distance(self.data.geometry[
            assignments == district_idx].unary_union.centroid) ** self.current_reduction_distance_bias).values
        neighbors = self.neighbors[assignments[self.neighbors['index_right']] == district_idx]['index_right']
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors.astype(np.float64) ** self.current_reduction_surrounding_bias).values
        selected, start = self._select_mutations(eligible, assignments, calculate_p(weights), centroid, start,
                                                 centroid_distance_weight=-1)
        if selected is None:
            return start

        neighbors = self.neighbors['index_right'][self.neighbors['index_right'] != district_idx]
        populations = self.data['population'].groupby(by=assignments).sum()
        centroids = {}
        for i in np.unique(assignments[neighbors]):
            mask = assignments == i
            geometries = self.data.geometry[mask]
            centroids[i] = self._calculate_union(mask, geometries).centroid
        district_selections, selection = [], None
        for x in selected:
            neighbor_districts = np.unique(assignments[neighbors[[x]]])
            if selection is None or selection not in neighbor_districts:
                neighbor_centroids = np.array([centroids[i] for i in neighbor_districts])
                weights = np.array([populations[i] for i in neighbor_districts]) ** \
                    self.current_expansion_population_bias * (
                    self.data.geometry[x].distance(neighbor_centroids) ** self.current_expansion_distance_bias)
                p = calculate_p(weights)
                selection = np.random.choice(neighbor_districts, p=p)
            district_selections.append(selection)

        assignments[selected] = district_selections
        return start

    def _mutation(self, assignments, district_idx):
        expand_start, reduce_start = None, None
        for _ in range(np.random.randint(low=max(self.mutation_layer_range[0], 1),
                                         high=max(self.mutation_layer_range[1], 1) + 1)):
            centroid = self._calculate_union(assignments == district_idx).centroid
            population_pct = self.data['population'][assignments == district_idx].sum() / self.ideal_pop
            p = calculate_p(np.array([population_pct ** self.current_expansion_population_bias, 1]))
            if np.random.random() < p[0]:
                expand_start = self._expansion(assignments, district_idx, centroid, expand_start)
            else:
                reduce_start = self._reduction(assignments, district_idx, centroid, reduce_start)

    def mutate(self, selected):
        population = copy.deepcopy(selected)
        for assignments in itertools.cycle(selected):
            assignments = assignments.copy()
            n_mutations = np.random.randint(low=max(int(self.current_mutation_n_range[0] * self.n_districts), 1),
                                            high=max(int(self.current_mutation_n_range[1] * self.n_districts), 1) + 1)
            districts = list(range(self.n_districts))
            np.random.shuffle(districts)
            for i, x in enumerate(itertools.cycle(districts), 1):
                self._mutation(assignments, x)
                if i == n_mutations:
                    break
            population.append(assignments)
            if len(population) == self.population_size:
                break
        return population

    def _plot(self, assignments):
        draw = self.live_plot and ('_temp_district' not in self.data or
                                   (self.data['_temp_district'] != assignments).any())
        save = self.save_dir is not None and self.save_every > 0 and self.generation % self.save_every == 0
        if not (save or draw):
            return
        self.data['_temp_district'] = assignments
        self.ax.clear()
        self.data.plot(column='_temp_district', ax=self.ax, cmap='tab20')
        if draw:
            plt.draw()
            plt.pause(0.10)
        if save:
            self.fig.savefig(f'{self.save_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}-{self.generation}')

    def time(self):
        return f'{dt.timedelta(seconds=round((dt.datetime.now() - self.start).total_seconds()))}'

    def log(self, message):
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')
        if self.verbose:
            print(message)

    def simulate_generation(self, last=False):
        self.log(f'{self.time()} - Generation: {self.generation:,} - Calculating fitness scores...')
        fitness_scores, metrics_list = self.calculate_fitness()
        self.log(f'{self.time()} - Generation: {self.generation:,} - Selecting best individuals...')
        selected, selected_metrics = self.select(fitness_scores, metrics_list)

        metric_strs = [f"{key}: {value}" for key, value in selected_metrics[0].items() if isinstance(value, str)]
        self.log(f'{self.time()} - Generation: {self.generation:,} - '
                 f'{" | ".join(metric_str for metric_str in metric_strs)}')

        self.log(f'{self.time()} - Generation: {self.generation:,} - Plotting best map...')
        self._plot(selected[0])

        if not last:
            self.log(f'{self.time()} - Generation: {self.generation:,} - Mutating for new generation...')
            self.population = self.mutate(selected)
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

    def run(self, generations=1):
        if self.live_plot:
            plt.ion()

        self.log(f'{self.time()} - Filling population...')
        self.fill_population(clear=True)

        self.log(f'{self.time()} - Simulating for {generations:,} generations...')
        for generation in range(generations + 1):
            self.simulate_generation(last=generation == generations)

        self.log(f'{self.time()} - Simulation complete!')
        if self.live_plot:
            plt.ioff()
            plt.show()


def main():
    start = dt.datetime.now()

    refresh_data = False
    if refresh_data:
        print(f'{time(start)} - Refreshing data...')
        data = gpd.read_file('data/pa/vtd-election-and-census-data-14-20.shp')
        data.to_crs(infer_utm_crs(data), inplace=True)
        data.rename(columns={'TOTPOP20': 'population', 'PRES20D': 'democrat', 'PRES20R': 'republican'}, inplace=True)
        data = tp.Topology(data, prequantize=False).toposimplify(50).to_gdf()
        data.geometry[~data.geometry.is_valid] = data.geometry[~data.geometry.is_valid].buffer(0)
        data.to_parquet('data/pa/simplified.parquet')

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingGeneticAlgorithm(
        data_path='data/pa/simplified.parquet',
        start=start,
        verbose=True,
        save_dir='maps',
        save_every=0,
        log_path='log.txt',
        live_plot=True,
        n_districts=17,
        weights={
            'pop_balance': 4,
            'compactness': 1,
            'competitiveness': 1,
            'efficiency_gap': 1,
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
        mutation_n_growth=17 ** (1 / 10_000),
        mutation_size_decay=0.01 ** (1 / 10_000),
        bias_decay=1.0,
    )

    algorithm.run(generations=20_000)


if __name__ == '__main__':
    main()
