import datetime as dt

import geopandas as gpd

from algorithm import since_start, DictCollection, ParameterCollection, Parameter, RangeParameter
from genetic_algorithm import GeneticRedistrictingAlgorithm
from maps import DistrictMap
from redistricting_env import refresh_data, RedistrictingEnv, infer_utm_crs


def calculate_fitness(district_map, weights):
    score = 0
    metrics = {'fitness': '0'}
    for metric, weight in weights.items():
        result = getattr(district_map, f'calculate_{metric}')()
        score += result * weight
        metrics[metric] = f'{result:.4%}'
    metrics['fitness'] = f'{score:.4f}'
    return score, metrics


def compare(districts, state, name):
    env = RedistrictingEnv(f'data/{state}/simplified.parquet', n_districts=districts, live_plot=False,
                           save_img_dir='maps')
    districts = gpd.read_file(f'data/{state}/current-boundaries.shp')
    solution = DistrictMap.load(f'maps/solutions/{state}/{name}.pkl', env=env)
    districts.to_crs(infer_utm_crs(districts), inplace=True)
    centroids = gpd.GeoDataFrame(env.data, geometry=env.data.geometry.centroid)
    assignments = gpd.sjoin(centroids, districts, how='left', predicate='within')['index_right'].values
    district_map = DistrictMap(env=env, assignments=assignments)
    district_map.save(f'maps/current/{state}')

    weights = DictCollection(
        contiguity=0,
        population_balance=-5,
        compactness=1,
        win_margin=-1,
        efficiency_gap=-1,
    )

    score, metrics = calculate_fitness(district_map, weights)
    metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in metrics.items()]
    print(f'Current District Metrics: {" | ".join(metric_str for metric_str in metric_strs)}')

    score, metrics = calculate_fitness(solution, weights)
    metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in metrics.items()]
    print(f'New Solution Metrics: {" | ".join(metric_str for metric_str in metric_strs)}')


def main():
    start = dt.datetime.now()

    state = 'pa'  # 'nc'
    districts = 17  # 14
    data_path = f'data/{state}/vtd-election-and-census.shp'
    simplified_path = f'data/{state}/simplified.parquet'

    refresh = False
    if refresh:
        print(f'{since_start(start)} - Refreshing data...')
        refresh_data(data_path=data_path, simplified_path=simplified_path)

    print(f'{since_start(start)} - Initiating algorithm...')
    algorithm = GeneticRedistrictingAlgorithm(
        env=RedistrictingEnv(
            data_path=simplified_path,
            n_districts=districts,
            live_plot=False,
            save_data_dir=f'maps/solutions/{state}',
            save_img_dir=None,
        ),
        starting_maps_dir=f'maps/random-starting-points/{state}',
        start=start,
        verbose=1,
        print_every=10,  # How often to log and print updates on the progress
        save_every=100,  # How often to save progress
        log_path='log.txt',
        population_size=2,  # For all practical purpose, a large population size will not provide valuable results
        selection_pct=0.5,  # Selects 1 from the population size of 2 and mutates that single map to generate another
        starting_population_size=25,  # Starts with a large population size in 0th generation and selects the best 2
        # -1 indicates that all maps found in the random-starting-points directory will be used to start the population
        weights=DictCollection(
            contiguity=0,  # Contiguity is not valued because
            population_balance=-5,  # Lower is better, most important factor (we want equal sized districts)
            compactness=1,  # Higher is better, indicates shapes are closer to a circle (perfectly compact)
            win_margin=-1,  # Lower is better, indicates districts are more competitive
            efficiency_gap=-1,  # Lower is better, discourages Gerrymandering from either side
        ),
        params=ParameterCollection(
            expansion_population_bias=Parameter(-0.6, exp_factor=1),  # The bias parameters aid the genetic mutations
            reduction_population_bias=Parameter(0.6, exp_factor=1),   # through heuristics that favor modifications
            expansion_distance_bias=Parameter(-0.2, exp_factor=1),    # that will likely result in increase fitness
            reduction_distance_bias=Parameter(0.2, exp_factor=1),
            expansion_surrounding_bias=Parameter(0.1, exp_factor=1),
            reduction_surrounding_bias=Parameter(-0.1, exp_factor=1),
            mutation_size=RangeParameter(0.0, 1.0, exp_factor=0.5 ** (1 / 10_000)),  # One mutation is a percentage of
            # additions or removals of all touching VTDs of a district
            mutation_layers=RangeParameter(1, 1, exp_factor=1 ** (1 / 20_000), min_value=1),  # More layers means
            # each mutation will include more layers of touching VTDs, allowing for larger changes
            mutation_n=RangeParameter(1 / districts, 1 / districts, exp_factor=2 ** (1 / 10_000), max_value=2),  # Higher "n" means
            # more modified districts per mutation, allowing for more complex changes
        ),
        min_p=0.1,  # Ensures that, despite heuristics, every mutation is still somewhat possible
    )

    # This will likely take several hours
    algorithm.run(generations=100_000)

    # Compare the current in place map to the new solution
    compare(districts=districts, state=state, name=algorithm.start.strftime("%Y-%m-%d-%H-%M-%S"))


if __name__ == '__main__':
    # compare(districts=17, state='pa', name='2024-03-22-03-16-37')
    main()
