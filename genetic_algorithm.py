import datetime as dt
import itertools

import geopandas as gpd
import pandas as pd
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
        print(geom.geom_type)
        return 0


def adaptive_topo_simplify(gdf, n_buckets, tolerance_pct):
    sqrt_areas = gdf['geometry'].area.apply(np.sqrt)

    min_area, max_area = sqrt_areas.min(), sqrt_areas.max()
    thresholds = np.linspace(min_area, max_area, n_buckets + 1)

    bucketed_gdfs = []
    for i in range(n_buckets):
        bucket_gdf = gdf[(sqrt_areas >= thresholds[i]) & (sqrt_areas < thresholds[i + 1])]
        tolerance = (thresholds[i] + thresholds[i + 1]) / 2 * tolerance_pct

        if not bucket_gdf.empty:
            simplified = tp.Topology(bucket_gdf, prequantize=False).toposimplify(tolerance).to_gdf()
            bucketed_gdfs.append(simplified)

    final_gdf = gpd.GeoDataFrame(pd.concat(bucketed_gdfs, ignore_index=True))
    return tp.Topology(final_gdf, prequantize=False).toposimplify(100).to_gdf()


class RedistrictingGeneticAlgorithm:
    def __init__(self, data_path, verbose=True, start=None, weights=None, population_size=100, selection_pct=0.5,
                 mutation_pct_range=(0.0, 0.10)):
        self.original = gpd.read_file(data_path)
        self.crs = infer_utm_crs(self.original)
        self.original = self.original.to_crs(self.crs)
        self.original.rename(columns={'P0010001': 'population'}, inplace=True)

        self.data = tp.Topology(self.original, prequantize=False).toposimplify(100).to_gdf()

        invalid_geoms = self.data['geometry'][~self.data['geometry'].is_valid]
        print(len(invalid_geoms), len(self.data), len(self.original))
        print(sum(count_points_in_geom(geom) for geom in self.data['geometry']) /
              sum(count_points_in_geom(geom) for geom in self.original['geometry']))

        # self.original.plot()
        # self.data.plot()
        # invalid_geoms.plot()
        # plt.show()

        self.total_pop = self.data['population'].sum()
        self.n_districts = 17
        self.ideal_pop = self.total_pop / self.n_districts

        self.weights = {
            'contiguity': 1,
            'pop_balance': 0,
            'compactness': 0,
        }
        for key in weights:
            if key in self.weights:
                self.weights[key] = weights[key]

        self.population_size = population_size
        self.selection_pct = selection_pct
        self.mutation_pct_range = mutation_pct_range

        self.generation = 0
        self.population = [self.random_map() for _ in range(population_size)]

        self.verbose = verbose
        if start is None:
            start = dt.datetime.now()
        self.start = start

    def random_map(self):
        return np.random.randint(low=0, high=self.n_districts, size=len(self.data))

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
            district_vtds = self.data[assignments == district_index]
            district_data.append({
                'district': district_index,
                'geometry': district_vtds.geometry[district_vtds.geometry.is_valid].unary_union
                if not district_vtds.empty else Polygon(np.zeros(6).reshape(3, 2)),
                'population': district_vtds['population'].sum()
            })

        return gpd.GeoDataFrame(district_data, crs=self.crs)

    def calculate_fitness(self):
        scores, best_score, metrics = {}, 0, {}
        for i, assignments in enumerate(self.population):
            district_map = self.construct_map(assignments)

            contiguity = self.calculate_contiguity(district_map)
            pop_balance = 0  # self.calculate_population_balance(district_map)
            compactness = 0  # self.calculate_compactness(district_map)

            fitness = contiguity * self.weights['contiguity'] + pop_balance * self.weights['pop_balance'] + \
                compactness * self.weights['compactness']
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
        for i in sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True):
            selected.append(self.population[i])

        return selected

    def mutate(self, selected):
        how_many = np.random.randint(low=max(self.mutation_pct_range[0] * len(self.data), 1),
                                     high=self.mutation_pct_range[1] * len(self.data),
                                     size=self.population_size - len(selected))

        population = selected.copy()
        for n, assignments in zip(how_many, itertools.cycle(selected)):
            assignments = assignments.copy()
            which = np.random.randint(low=0, high=len(self.data), size=n)
            what = np.random.randint(low=0, high=self.n_districts, size=n)
            assignments[which] = what
            population.append(assignments)

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


def main():
    start = dt.datetime.now()

    verbose = True
    weights = {}
    population_size = 100
    selection_pct = 0.5
    mutation_pct_range = (0.0, 0.10)
    generations = 100

    print(f'{time(start)} - Initiating algorithm...')
    algorithm = RedistrictingGeneticAlgorithm('data/pa/WP_VotingDistricts.shp', verbose=verbose, start=start,
                                              weights=weights, population_size=population_size,
                                              selection_pct=selection_pct, mutation_pct_range=mutation_pct_range)

    print(f'{time(start)} - Simulating for {generations} generations...')
    algorithm.run(generations=generations)


if __name__ == '__main__':
    main()
