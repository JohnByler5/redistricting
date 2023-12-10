import datetime as dt
import itertools
import copy

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import topojson as tp

import matplotlib.pyplot as plt


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
    def __init__(self, data_path, verbose=True, start=None, weights=None, population_size=100, selection_pct=0.5,
                 mutation_n_range=(0.0, 1.0), mutation_size_range=(0.0, 0.50)):
        self.original_data = gpd.read_file(data_path)
        self.crs = infer_utm_crs(self.original_data)
        self.original_data = self.original_data.to_crs(self.crs)
        self.original_data.rename(columns={'P0010001': 'population'}, inplace=True)

        self.data = tp.Topology(self.original_data, prequantize=False).toposimplify(100).to_gdf()
        self.data.geometry[~self.data.geometry.is_valid] = self.data.geometry[~self.data.geometry.is_valid].buffer(0)

        # invalid_geoms = self.data[~self.data.geometry.is_valid]
        # print(len(invalid_geoms), len(self.data), len(self.original_data))
        # print(sum(count_points_in_geom(geom) for geom in self.data.geometry) /
        #       sum(count_points_in_geom(geom) for geom in self.original_data.geometry))
        #
        # self.original_data.plot()
        # self.data.plot()
        # plt.show()

        _ = self.data.sindex  # "_ = " is just to get rid of the PyCharm notice, but it is unnecessary
        self.neighbors = gpd.sjoin(self.data, self.data, how='inner', predicate='touches')
        self.neighbors = self.neighbors[self.neighbors.index != self.neighbors['index_right']]

        self.total_pop = self.original_data['population'].sum()
        self.n_districts = 17
        self.ideal_pop = self.total_pop / self.n_districts

        self.verbose = verbose
        if start is None:
            start = dt.datetime.now()
        self.start = start

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

        self.generation = 0
        print(f'{time(start)} - Generating random maps')
        self.pop_count = 0
        self.population = [self.random_map() for _ in range(population_size)]
        print(f'{time(start)} - Done')

    def random_map(self):
        self.pop_count += 1
        print(f'{time(self.start)} {self.pop_count}')
        assignments = np.full(len(self.original_data), -1)

        starting_points = np.random.choice(len(self.original_data), self.n_districts, replace=False)
        for district, starting_point in enumerate(starting_points):
            assignments[starting_point] = district

        just_geometry = self.data[['geometry']]
        skipped_districts, n_allocated = set(), self.n_districts
        while n_allocated < len(self.original_data) and len(skipped_districts) != self.n_districts:
            districts = np.array(range(self.n_districts))
            np.random.shuffle(districts)
            for district in districts:
                if district in skipped_districts:
                    continue

                touching_vtds = np.unique(gpd.sjoin(just_geometry[assignments == -1],
                                                    just_geometry[assignments == district],
                                                    how='inner', predicate='touches').index.values)

                if not len(touching_vtds):
                    skipped_districts.add(district)
                    continue

                assignments[touching_vtds] = district
                n_allocated += len(touching_vtds)

        while n_allocated < len(self.original_data):
            for i in np.where(assignments == -1)[0]:
                unallocated_vtd = self.data.geometry.iloc[i]

                results = self.data.geometry[assignments != -1].sindex.nearest(unallocated_vtd, return_all=False,
                                                                               exclusive=True).flatten()

                min_distance, closest_vtd = None, results[0]
                for vtd in results:
                    distance = unallocated_vtd.distance(self.data.geometry.iloc[vtd])
                    if min_distance is None or distance < min_distance:
                        min_distance, closest_vtd = distance, vtd

                if assignments[closest_vtd] == -1:
                    continue

                assignments[i] = assignments[closest_vtd]
                n_allocated += 1

        return assignments

    def calculate_contiguity(self, district_map):
        total_breaks = sum(district_map['geometry'].apply(count_polygons) - 1)
        max_breaks = len(self.original_data) - self.n_districts
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

        return gpd.GeoDataFrame(district_data, crs=self.crs)

    def calculate_fitness(self):
        scores, best_score, metrics = {}, 0, {}
        for i, assignments in enumerate(self.population):
            district_map = self.construct_map(assignments)

            contiguity = self.calculate_contiguity(district_map)
            pop_balance = self.calculate_population_balance(district_map)
            compactness = self.calculate_compactness(district_map)

            fitness = pop_balance * self.weights['pop_balance'] + compactness * self.weights['compactness']
            scores[i] = fitness

            if fitness > best_score:
                best_score = fitness
                metrics = {
                    'Fitness': f'{fitness:.4f}',
                    'Contiguity': f'{contiguity:.4%}',
                    'Population Balance': f'{pop_balance:.4%}',
                    'Compactness': f'{compactness:.4%}',
                }

        return scores, metrics

    def select(self, fitness_scores):
        selected = []
        n = min(max(round(self.population_size * self.selection_pct), 1), self.population_size - 1)
        for i in sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:n]:
            selected.append(self.population[i])

        return selected

    def get_best(self):
        fitness_scores, metrics = self.calculate_fitness()
        return self.construct_map(self.select(fitness_scores)[0])

    def mutate(self, selected):
        population = copy.deepcopy(selected)

        for assignments in itertools.cycle(selected):
            new_assignments = assignments.copy()
            for _ in range(np.random.randint(low=max(int(self.mutation_n_range[0] * self.n_districts), 1),
                                             high=max(int(self.mutation_n_range[1] * self.n_districts), 1) + 1)):

                district_idx = np.random.randint(low=0, high=self.n_districts)

                eligible_vtds = self.neighbors[new_assignments[self.neighbors.index] == district_idx]
                eligible_vtds = eligible_vtds[new_assignments[eligible_vtds.index] != new_assignments[
                    eligible_vtds['index_right']]]['index_right'].drop_duplicates()

                if eligible_vtds.empty:
                    continue

                size = np.random.randint(low=max(int(self.mutation_size_range[0] * len(eligible_vtds)), 1),
                                         high=max(int(self.mutation_size_range[1] * len(eligible_vtds)), 1) + 1)
                starting_vtd = np.random.choice(eligible_vtds)

                processed_vtds = {starting_vtd}
                last_additions = [starting_vtd]

                while len(eligible_vtds) > 0 and len(processed_vtds) < size:
                    eligible_vtds = eligible_vtds[~eligible_vtds.isin(processed_vtds)]

                    touching_neighbors = eligible_vtds.values[self.data.geometry.iloc[eligible_vtds].touches(
                        self.data.geometry.iloc[last_additions].unary_union)]
                    if len(touching_neighbors) == 0:
                        break

                    if len(touching_neighbors) > size - len(processed_vtds):
                        np.random.shuffle(touching_neighbors)
                        touching_neighbors = touching_neighbors[:size - len(processed_vtds)]

                    processed_vtds.update(touching_neighbors)

                    all_neighbors = self.data.geometry.iloc[self.neighbors["index_right"][list(processed_vtds)]]
                    if not isinstance(all_neighbors.unary_union, Polygon):
                        processed_vtds.difference_update(touching_neighbors)
                        break

                    last_additions = touching_neighbors

                new_assignments[list(processed_vtds)] = district_idx

            population.append(new_assignments)

            # data = self.data.copy()
            # data['district'] = assignments
            # data.plot('district')
            # data['district'] = new_assignments
            # data.plot('district')
            # data['new'] = data.index.isin(processed_vtds)
            # data.plot('new')
            # plt.show()

            if len(population) == self.population_size:
                break

        return population

    def simulate_generation(self):
        self.generation += 1
        fitness_scores, metrics = self.calculate_fitness()

        if self.verbose:
            print(f'{time(self.start)} - Generation: {self.generation:,} | '
                  f'{" | ".join(f"{key}: {value}" for key, value in metrics.items())}')

        selected = self.select(fitness_scores)
        self.population = self.mutate(selected)

    def run(self, generations=1):
        for _ in range(generations):
            self.simulate_generation()

        best = self.get_best()
        best.plot()
        plt.show()


def main():
    start = dt.datetime.now()

    verbose = True
    weights = {}
    population_size = 100
    selection_pct = 0.05
    mutation_n_range = (0.0, 1.0)
    mutation_size_range = (0.0, 0.1)
    generations = 1000

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingGeneticAlgorithm('data/pa/WP_VotingDistricts.shp', verbose=verbose, start=start,
                                              weights=weights, population_size=population_size,
                                              selection_pct=selection_pct, mutation_n_range=mutation_n_range,
                                              mutation_size_range=mutation_size_range)

    print(f'{time(start)} - Simulating for {generations} generations...')
    algorithm.run(generations=generations)


if __name__ == '__main__':
    main()
