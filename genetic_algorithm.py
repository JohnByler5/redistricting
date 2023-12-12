import copy
import datetime as dt
import itertools

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import topojson as tp
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString


def count_polygons(geometry):
    if isinstance(geometry, Polygon):
        return 1
    elif isinstance(geometry, MultiPolygon):
        return len(geometry.geoms)
    else:
        raise ValueError(f'Must be instance of "Polygon" or "MultiPolygon", not "{type(geometry)}"')


def time(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


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
    elif isinstance(geom, (MultiPolygon, MultiLineString)):
        return sum(count_points_in_geom(part) for part in geom.geoms)
    else:
        raise AssertionError(f'Incorrect geom type: {type(geom)}')


class RedistrictingGeneticAlgorithm:
    def __init__(
            self,
            data_path,
            start=dt.datetime.now(),
            verbose=True,
            save_dir=None,
            save_every=1,
            weights=None,
            population_size=100,
            selection_pct=0.5,
            mutation_n_range=(0.0, 1.0),
            mutation_size_range=(0.0, 0.50),
            mutation_population_bias=0,
            starting_population_size=None,
    ):
        if selection_pct > 0.5:
            raise ValueError('Parameter `selection_pct` must be less than or equal to `0.5`.')

        self.data = gpd.read_parquet(data_path)

        _ = self.data.sindex  # "_ = " is just to get rid of the PyCharm notice, but it is unnecessary
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
        self.mutation_population_bias = mutation_population_bias
        if starting_population_size is None:
            starting_population_size = population_size
        self.starting_population_size = starting_population_size

        self.generation = 0
        self.pop_count = 0
        print(f'{time(start)} - Filling population...')
        self.population = [self.random_map() for _ in range(starting_population_size)]

    def random_map(self):
        self.pop_count += 1
        assignments = np.full(len(self.data), -1)

        starting_points = np.random.choice(len(self.data), self.n_districts, replace=False)
        for district, starting_point in enumerate(starting_points):
            assignments[starting_point] = district

        skipped_districts, n_allocated = set(), self.n_districts
        index_mask = self.data.index.isin(self.neighbors.index)
        while n_allocated < len(self.data) and len(skipped_districts) != self.n_districts:
            districts = np.array(range(self.n_districts))
            np.random.shuffle(districts)
            for district in districts:
                if district in skipped_districts:
                    continue

                indices = self.data.index[(assignments == district) & index_mask]
                touching_vtds = self.neighbors["index_right"][indices].unique()
                touching_vtds = touching_vtds[assignments[touching_vtds] == -1]

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

    def construct_map(self, assignments):
        district_data = []
        for district_index in range(self.n_districts):
            vtds = self.data[assignments == district_index]

            district_data.append({
                'district': district_index,
                'geometry': vtds.geometry.unary_union if not vtds.empty else Polygon(np.zeros(6).reshape(3, 2)),
                'population': vtds['population'].sum()
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

    def mutate(self, selected):
        population = copy.deepcopy(selected)

        for assignments in itertools.cycle(selected):
            new_assignments = assignments.copy()
            change = np.zeros(assignments.shape)

            n_mutations = np.random.randint(low=max(int(self.mutation_n_range[0] * self.n_districts), 1),
                                            high=max(int(self.mutation_n_range[1] * self.n_districts), 1) + 1)
            sizes = {}
            for m in range(n_mutations):

                p = self.data['population'].groupby(by=new_assignments).sum()
                p = p.values ** self.mutation_population_bias
                p /= p.sum()
                district_idx = np.random.choice(np.array(range(self.n_districts)), p=p)

                eligible_vtds = self.neighbors[new_assignments[self.neighbors.index] == district_idx]
                eligible_vtds = eligible_vtds[new_assignments[eligible_vtds.index] != new_assignments[
                    eligible_vtds['index_right']]]['index_right'].drop_duplicates()

                if eligible_vtds.empty:
                    continue

                size = np.random.randint(low=max(int(self.mutation_size_range[0] * len(eligible_vtds)), 1),
                                         high=max(int(self.mutation_size_range[1] * len(eligible_vtds)), 1) + 1)
                sizes[size] = 0

                attempts = 0
                while True:
                    starting_vtd = np.random.choice(eligible_vtds)

                    for i in range(self.n_districts):
                        if i == district_idx:
                            continue
                        before = self.data.geometry[new_assignments == i]
                        after = before[before.index != starting_vtd]
                        if count_polygons(after.unary_union) > count_polygons(before.unary_union):
                            break
                    else:
                        break
                    attempts += 1
                    if attempts == 10 or attempts >= len(eligible_vtds):
                        break
                if attempts == 10 or attempts >= len(eligible_vtds):
                    continue

                eligible_vtds.sort_values(key=lambda x: self.data.geometry[x].distance(
                    self.data.geometry[starting_vtd]), inplace=True)
                eligible_vtds = eligible_vtds.values[:size]

                selected, low, high = eligible_vtds.copy(), 0, len(eligible_vtds)
                while True:
                    if low + 1 >= high:
                        break

                    breaks_district = False
                    for i in np.unique(new_assignments[selected]):
                        if i == district_idx:
                            continue

                        before = self.data.geometry[new_assignments == i]
                        after = before[~before.index.isin(selected)]
                        if count_polygons(after.unary_union) > count_polygons(before.unary_union):
                            breaks_district = True
                            break

                    if breaks_district:
                        high = len(selected)
                    else:
                        low = len(selected)
                    selected = eligible_vtds[:(low + high) // 2]

                for i in np.unique(new_assignments[selected]):
                    if i == district_idx:
                        continue
                    before = self.data.geometry[new_assignments == i]
                    after = before[~before.index.isin(selected)]
                    if count_polygons(after.unary_union) > count_polygons(before.unary_union):
                        print('Contiguity system failed!')

                sizes[size] = len(selected)
                new_assignments[selected] = district_idx
                change[selected] = m + 1

            # print(n_mutations, sizes)
            # to_plot = self.data.copy()
            # to_plot['old_district'] = assignments
            # to_plot['new_district'] = new_assignments
            # to_plot['change'] = change
            # to_plot.plot('old_district')
            # to_plot.plot('new_district')
            # to_plot.plot('change')
            # plt.show()

            population.append(new_assignments)
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

    def simulate_generation(self):
        print(f'{time(self.start)} - Generation: {self.generation:,} - Calculating fitness scores...')
        fitness_scores, metrics_list = self.calculate_fitness()
        print(f'{time(self.start)} - Generation: {self.generation:,} - Selecting best individuals...')
        selected, selected_metrics = self.select(fitness_scores, metrics_list)

        if self.verbose:
            metric_strs = [f"{key}: {value}" for key, value in selected_metrics[0].items() if isinstance(value, str)]
            print(f'{time(self.start)} - Generation: {self.generation:,} - '
                  f'{" | ".join(metric_str for metric_str in metric_strs)}')

        if self.save_dir is not None and self.generation % self.save_every == 0:
            print(f'{time(self.start)} - Generation: {self.generation} - Plotting best map...')
            self.plot(selected[0])

        print(f'{time(self.start)} - Generation: {self.generation:,} - Mutating for new generation...')
        self.population = self.mutate(selected)
        self.generation += 1

    def run(self, generations=1):
        if self.verbose:
            print(f'{time(self.start)} - Simulating for {generations} generations...')

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
            'pop_balance': 5,
            'compactness': 1,
        },
        population_size=5,
        selection_pct=0.2,
        mutation_n_range=(0.0, 1.0),
        mutation_size_range=(0.0, 1.0),
        mutation_population_bias=-10,
        starting_population_size=5,
    )

    algorithm.run(generations=1_000)


if __name__ == '__main__':
    main()
