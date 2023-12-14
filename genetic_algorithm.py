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
            weights=None,
            population_size=100,
            selection_pct=0.5,
            mutation_n_range=(0.0, 1.0),
            mutation_size_range=(0.0, 0.50),
            expansion_population_bias=0,
            reduction_population_bias=0,
            expansion_distance_bias=0,
            reduction_distance_bias=0,
            expansion_surrounding_bias=0,
            reduction_surrounding_bias=0,
            starting_population_size=None,
            mutation_size_decay=1.0,
            use_reduction=True,
    ):
        if selection_pct > 0.5:
            raise ValueError('Parameter `selection_pct` must be less than or equal to `0.5`.')

        self.data = gpd.read_parquet(data_path)

        self.neighbors = gpd.sjoin(self.data, self.data, how='inner', predicate='touches')
        self.neighbors = self.neighbors[self.neighbors.index != self.neighbors['index_right']]

        self.total_pop = self.data['population'].sum()
        self.n_districts = 17
        self.ideal_pop = self.total_pop / self.n_districts

        self.verbose = verbose
        self.start = start
        self.save_dir = save_dir
        self.save_every = save_every

        self.weights = {
            'pop_balance': 1,
            'compactness': 1,
        }
        for key in weights:
            if key in self.weights:
                self.weights[key] = weights[key]

        self.population_size = population_size
        self.selection_pct = selection_pct
        self.mutation_n_range = mutation_n_range
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
        self.mutation_size_decay = mutation_size_decay
        self.use_reduction = use_reduction

        self.generation = 0
        print(f'{time(start)} - Filling population...')
        self.population = [self.random_map() for _ in range(starting_population_size)]
        self.current_mutation_size_range = mutation_size_range

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

    def random_map(self):
        assignments = np.full(len(self.data), -1)

        starting_points = np.random.choice(len(self.data), self.n_districts, replace=False)
        for district, starting_point in enumerate(starting_points):
            assignments[starting_point] = district

        skipped_districts, n_allocated = set(), self.n_districts
        neighbors = self.neighbors['index_right']
        while n_allocated < len(self.data) and len(skipped_districts) != self.n_districts:
            districts = np.array(range(self.n_districts))
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
                unallocated_vtd = self.data.geometry[i]

                results = self.data.geometry[assignments != -1].sindex.nearest(unallocated_vtd, return_all=False,
                                                                               exclusive=True).flatten()

                min_distance, closest_vtd = None, results[0]
                for vtd in results:
                    distance = unallocated_vtd.distance(self.data.geometry[vtd])
                    if min_distance is None or distance < min_distance:
                        min_distance, closest_vtd = distance, vtd

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
        total_difference = (district_map['population'] - self.ideal_pop).abs().sum()
        max_difference = self.total_pop * (self.n_districts - 1) / self.n_districts * 2
        return 1 - total_difference / max_difference

    @staticmethod
    def calculate_compactness(district_map):
        return (4 * np.pi * district_map.geometry.area / district_map.geometry.length ** 2).mean()

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
                'population': self.data['population'][mask].sum()
            })

        return gpd.GeoDataFrame(district_data, crs=self.data.crs)

    def calculate_fitness(self):
        scores, metrics_list = {}, []
        for i, assignments in enumerate(self.population):
            district_map = self.construct_map(assignments)

            contiguity = self.calculate_contiguity(district_map)
            pop_balance = self.calculate_population_balance(district_map)
            compactness = self.calculate_compactness(district_map)

            fitness = pop_balance * self.weights['pop_balance'] + compactness * self.weights['compactness']
            scores[i] = fitness

            metrics_list.append({
                'Fitness': f'{fitness:.4f}',
                'Contiguity': f'{contiguity:.4%}',
                'Population Balance': f'{pop_balance:.4%}',
                'Compactness': f'{compactness:.4%}',
                'pop_balance_by_district': district_map['population'] / self.ideal_pop
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
        attempts = 0
        while attempts < len(eligible):
            i = np.random.choice(range(len(eligible)), p=p)
            x = eligible.iloc[i]
            if self._check_contiguity(previous_unions, assignments[x], assignments, [x]):
                return x
            p[i] = 0
            p = calculate_p(p)
            attempts += 1
        return None

    def _select_mutations(self, eligible, assignments, p):
        previous_unions = {}
        start = self._choose_start(eligible, assignments, previous_unions, p)
        if start is None:
            return None
        eligible = eligible.sort_values(key=lambda x: self.data.geometry[x].distance(
            self.data.geometry[start])).values[:self._rand_mutation_size(len(eligible))]
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
        return selected

    def _get_border(self, assignments, which, reverse=False):
        if reverse:
            i1, i2 = self.neighbors['index_right'], self.neighbors.index
        else:
            i1, i2 = self.neighbors.index, self.neighbors['index_right']
        return self.neighbors[(assignments[i1] == which) & (assignments[i2] != which)]['index_right'].drop_duplicates()

    def _expansion(self, assignments, district_idx):
        eligible = self._get_border(assignments, district_idx)
        if eligible.empty:
            return
        weights = (self.data['population'].groupby(by=assignments).sum()[
            assignments[eligible]] ** self.reduction_population_bias).values
        weights *= (self.data.geometry[eligible].distance(self.data.geometry[
            assignments == district_idx].unary_union.centroid) ** self.expansion_distance_bias).values
        neighbors = self.neighbors[assignments[self.neighbors['index_right']] == district_idx]['index_right']
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors ** self.expansion_surrounding_bias).values
        selected = self._select_mutations(eligible, assignments, calculate_p(weights))
        if selected is None:
            return
        assignments[selected] = district_idx

    def _reduction(self, assignments, district_idx):
        eligible = self._get_border(assignments, district_idx, reverse=True)
        if eligible.empty:
            return
        weights = (self.data.geometry[eligible].distance(self.data.geometry[
            assignments == district_idx].unary_union.centroid) ** self.reduction_distance_bias).values
        neighbors = self.neighbors[assignments[self.neighbors['index_right']] == district_idx]['index_right']
        neighbors = neighbors[eligible[eligible.isin(neighbors.index)]]
        neighbors = neighbors.groupby(by=neighbors.index).apply(len).reindex(eligible, fill_value=0) + 1
        weights *= (neighbors.astype(float) ** self.reduction_surrounding_bias).values
        selected = self._select_mutations(eligible, assignments, calculate_p(weights))
        if selected is None:
            return

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
                weights = np.array([populations[i] for i in neighbor_districts]) ** self.expansion_population_bias * (
                    self.data.geometry[x].distance(neighbor_centroids) ** self.expansion_distance_bias)
                p = calculate_p(weights)
                selection = np.random.choice(neighbor_districts, p=p)
            district_selections.append(selection)

        assignments[selected] = district_selections

    def _mutation(self, assignments):
        district_idx = np.random.choice(np.array(range(self.n_districts)))
        population_pct = self.data['population'][assignments == district_idx].sum() / self.ideal_pop
        p = calculate_p(np.array([population_pct ** self.expansion_population_bias, 1]))
        if (not self.use_reduction) or (np.random.random() < p[0]):
            self._expansion(assignments, district_idx)
        else:
            self._reduction(assignments, district_idx)

    def mutate(self, selected):
        population = copy.deepcopy(selected)
        for assignments in itertools.cycle(selected):
            assignments = assignments.copy()
            n_mutations = np.random.randint(low=max(int(self.mutation_n_range[0] * self.n_districts), 1),
                                            high=max(int(self.mutation_n_range[1] * self.n_districts), 1) + 1)
            for m in range(n_mutations):
                self._mutation(assignments)
            population.append(assignments)
            if len(population) == self.population_size:
                break
        return population

    def plot(self, best_assignments):
        to_plot = self.data.copy()
        to_plot['district'] = best_assignments
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        to_plot.plot(column='district', ax=ax, cmap='tab20')
        ax.set_title('Congressional Districts')
        plt.savefig(f'{self.save_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}-{self.generation}')
        plt.close(fig)

    def time(self):
        return f'{dt.timedelta(seconds=round((dt.datetime.now() - self.start).total_seconds()))}'

    def simulate_generation(self):
        print(f'{self.time()} - Generation: {self.generation:,} - Calculating fitness scores...')
        fitness_scores, metrics_list = self.calculate_fitness()
        print(f'{self.time()} - Generation: {self.generation:,} - Selecting best individuals...')
        selected, selected_metrics = self.select(fitness_scores, metrics_list)

        if self.verbose >= 1:
            metric_strs = [f"{key}: {value}" for key, value in selected_metrics[0].items() if isinstance(value, str)]
            print(f'{self.time()} - Generation: {self.generation:,} - '
                  f'{" | ".join(metric_str for metric_str in metric_strs)}')

        if self.save_dir is not None and self.generation % self.save_every == 0:
            print(f'{self.time()} - Generation: {self.generation} - Plotting best map...')
            self.plot(selected[0])

        print(f'{self.time()} - Generation: {self.generation:,} - Mutating for new generation...')
        self.population = self.mutate(selected)
        self.generation += 1

        self.current_mutation_size_range = (
            self.current_mutation_size_range[0] * self.mutation_size_decay,
            self.current_mutation_size_range[1] * self.mutation_size_decay,
        )

    def run(self, generations=1):
        if self.verbose >= 1:
            print(f'{self.time()} - Simulating for {generations} generations...')

        for generation in range(generations):
            self.simulate_generation()


def main():
    start = dt.datetime.now()

    refresh_data = False
    if refresh_data:
        print(f'{time(start)} - Refreshing data...')
        data = gpd.read_file('data/pa/WP_VotingDistricts.shp')
        data = data.to_crs(infer_utm_crs(data))
        data.rename(columns={'P0010001': 'population'}, inplace=True)
        data = tp.Topology(data, prequantize=False).toposimplify(100).to_gdf()
        data.geometry[~data.geometry.is_valid] = data.geometry[~data.geometry.is_valid].buffer(0)
        data.to_parquet('data/pa/simplified.parquet')

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingGeneticAlgorithm(
        data_path='data/pa/simplified.parquet',
        start=start,
        verbose=True,
        save_dir='maps',
        save_every=10,
        weights={
            'pop_balance': 1,
            'compactness': 3,
        },
        population_size=25,
        selection_pct=0.2,
        mutation_n_range=(0.0, 1.0),
        mutation_size_range=(0.0, 1.0),
        expansion_population_bias=-10,
        reduction_population_bias=10,
        expansion_distance_bias=-10,
        reduction_distance_bias=10,
        expansion_surrounding_bias=2,
        reduction_surrounding_bias=-2,
        starting_population_size=1_000,
        mutation_size_decay=0.8 ** (1 / 100),
        use_reduction=True,
    )

    algorithm.run(generations=3_000)


if __name__ == '__main__':
    main()
