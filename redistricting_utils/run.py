import geopandas as gpd

from .algorithm import DictCollection, ParameterCollection, Parameter, RangeParameter
from .genetic_algorithm import GeneticRedistrictingAlgorithm
from .maps import DistrictMap
from .env import RedistrictingEnv, infer_utm_crs


def refresh_current_maps(states, config):
    for state in states:
        env = RedistrictingEnv(
            data_path=config['states'][state]['paths']['simplified_raw_data'],
            state=state,
            n_districts=config['states'][state]['n_districts'],
            live_plot=False,
            save_data_dir=config['states'][state]['paths']['solution_data_dir'],
            save_img_dir=config['states'][state]['paths']['solution_images_dir'],
        )

        districts = gpd.read_file(config['states'][state]['paths']['current_boundaries'])
        districts.to_crs(infer_utm_crs(districts), inplace=True)
        centroids = gpd.GeoDataFrame(env.data, geometry=env.data.geometry.centroid)
        assignments = gpd.sjoin(centroids, districts, how='left', predicate='within')['index_right'].values
        district_map = DistrictMap(env=env, assignments=assignments)
        district_map.save(config['states'][state]['paths']['current_data'])
        district_map.plot(config['states'][state]['paths']['current_image'])


def compare(state, name, weights, config):
    env = RedistrictingEnv(data_path=config['states'][state]['paths']['simplified_raw_data'], state=state,
                           n_districts=config['states'][state]['n_districts'], live_plot=False,
                           save_img_dir=config['states'][state]['paths']['solution_images_dir'])
    current = DistrictMap.load(config['states'][state]['paths']['current_data'], env=env)
    solution = DistrictMap.load(f'{config["states"][state]["paths"]["solution_data_dir"]}/{name}', env=env)

    score, metrics = current.calculate_fitness(weights)
    metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in metrics.items()]
    print(f'Current District Metrics: {" | ".join(metric_str for metric_str in metric_strs)}')

    score, metrics = solution.calculate_fitness(weights)
    metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in metrics.items()]
    print(f'New Solution Metrics: {" | ".join(metric_str for metric_str in metric_strs)}')


def create_algorithm(config):
    state = 'pa'

    env = RedistrictingEnv(
        state=state,
        n_districts=config['states'][state]['n_districts'],
        data_path=config['states'][state]['paths']['simplified'],
        current_path=config['states'][state]['paths']['current_data'],
        save_data_dir=config['states'][state]['paths']['solution_data_dir'],
        save_img_dir=config['states'][state]['paths']['solution_images_dir'],
        live_plot=False,
    )

    weights = DictCollection(
        contiguity=0,
        population_balance=-5,
        compactness=1,
        win_margin=-1,
        efficiency_gap=-1,
    )

    params = ParameterCollection(
        expansion_population_bias=Parameter(-0.6, exp_factor=1),  # The bias parameters aid the genetic mutations
        reduction_population_bias=Parameter(0.6, exp_factor=1),  # through heuristics that favor modifications
        expansion_distance_bias=Parameter(-0.2, exp_factor=1),  # that will likely result in increase fitness
        reduction_distance_bias=Parameter(0.2, exp_factor=1),
        expansion_surrounding_bias=Parameter(0.1, exp_factor=1),
        reduction_surrounding_bias=Parameter(-0.1, exp_factor=1),
        mutation_size=RangeParameter(0.0, 1.0, exp_factor=0.5 ** (1 / 10_000)),  # One mutation is a percentage of
        # additions or removals of all touching VTDs of a district
        mutation_layers=RangeParameter(1, 1, exp_factor=1 ** (1 / 20_000), min_value=1),  # More layers means
        # each mutation will include more layers of touching VTDs, allowing for larger changes
        mutation_n=RangeParameter(1 / env.n_districts, 1 / env.n_districts, exp_factor=2 ** (1 / 10_000), max_value=2),
        # Higher "n" means more modified districts per mutation, allowing for more complex changes
    )

    algorithm = GeneticRedistrictingAlgorithm(
        env=env,
        starting_maps_dir=config['states'][state]['paths']['starting_maps_dir'],
        verbose=1,
        print_every=10,  # How often to log and print updates on the progress
        save_every=10,  # How often to save progress
        log_path=config['log_path'],
        population_size=2,  # For all practical purpose, a large population size will not provide valuable results
        selection_pct=0.5,  # Selects 1 from the population size of 2 and mutates that single map to generate another
        starting_population_size=25,  # Starts with a large population size in 0th generation and selects the best 2
        # -1 indicates that all maps found in the random-starting-points directory will be used to start the population
        weights=weights,
        params=params,
        min_p=0.1,  # Ensures that, despite heuristics, every mutation is still somewhat possible
    )

    return algorithm
