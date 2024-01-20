import copy
import datetime as dt
import os

from algorithm import Algorithm, Parameter, RangeParameter, ParameterCollection, DictCollection
from genetic_algorithm import GeneticRedistrictingAlgorithm
from maps import DistrictMap, DistrictMapCollection
from redistricting_env import RedistrictingEnv


def save_random_maps(env, weights, start_n, save_n, save_dir='maps/random-starting-points'):
    assert start_n >= 1
    assert save_n >= 1
    assert start_n >= save_n
    print(f'Generating {start_n:,} maps...')
    collection = DistrictMapCollection(env=env, max_size=start_n)
    collection.randomize()
    print('Calculating fitness scores...')
    fitness_scores, _ = collection.calculate_fitness(weights=weights)
    print(f'Selecting best {save_n:,} maps...')
    indices = sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:save_n]
    selected = collection.select(indices, new_max_size=save_n)
    print('Saving selected maps...')
    dt_str = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(save_dir, dt_str)
    os.mkdir(save_dir)
    for i, map_ in enumerate(selected, 1):
        save_path = f'{os.path.join(save_dir, str(i))}.pkl'
        map_.save(save_path)


class AlgorithmTest:
    def __init__(self, algorithm: type[Algorithm], param_sets: dict, test_size: int, start_map_path=None):
        self.algorithm = algorithm
        self.param_sets = param_sets
        self.test_size = test_size
        self.start_map_path = start_map_path
        self.log = algorithm(**list(param_sets.values())[0]).log

    def run(self, generations, results_path):
        self.log(f'Starting test with {len(self.param_sets)} param sets, test_size={self.test_size}, and '
                 f'generations={generations}...')
        results = {}
        for value, params in self.param_sets.items():
            algorithms = [self.algorithm(**params) for _ in range(self.test_size)]
            if self.start_map_path is not None:
                files = os.listdir(self.start_map_path)
                assert len(files) >= self.test_size, 'Start map path does not have enough files'
                for i, alg in enumerate(algorithms):
                    map_ = DistrictMap.load(os.path.join(self.start_map_path, files[i]))
                    alg.set_start_map(map_)

            self.log(f'Starting batch with value={value}...', verbose=0)
            fitness_scores, improvements = [], []
            for i, algorithm in enumerate(algorithms, 1):
                fitness_history = algorithm.run(generations)
                self.log(f'({i}/{len(algorithms)}) - Fitness: {fitness_history[-1]}', verbose=0)
                fitness_scores.append(fitness_history[-1])
                improvements.append(fitness_history[-1] - fitness_history[0])
            self.log(f'Batch complete!\nValue: {value} - Min Fitness: {min(fitness_scores):.4f} | '
                     f'Avg Fitness: {sum(fitness_scores) / self.test_size:.4f} | Max Fitness: {max(fitness_scores):.4f} | '
                     f'Min Improvement: {min(improvements):.4f} | '
                     f'Avg Improvement: {sum(improvements) / self.test_size:.4f} | '
                     f'Max Improvement: {max(improvements):.4f}', verbose=0)
            results[value] = (fitness_scores, improvements)

        results_str = ''
        for i, (value, (fitness_scores, improvements)) in enumerate(results.items(), 1):
            results_str += f'{value} - Min Fitness: {min(fitness_scores):.4f} | ' \
                           f'Avg Fitness: {sum(fitness_scores) / self.test_size:.4f} | ' \
                           f'Max Fitness: {max(fitness_scores):.4f} | Min Improvement: {min(improvements):.4f} | ' \
                           f'Avg Improvement: {sum(improvements) / self.test_size:.4f} | ' \
                           f'Max Improvement: {max(improvements):.4f}'
            if i < len(results):
                results_str += '\n'
        with open(results_path, 'w') as f:
            f.write(results_str)

def main():
    env = RedistrictingEnv(
        data_path='data/pa/simplified.parquet',
        n_districts=17,
        live_plot=False,
        save_data_dir=None,
        save_img_dir=None,
    )
    weights = DictCollection(
        contiguity=0,
        population_balance=5,
        compactness=1,
        win_margin=-1,
        efficiency_gap=-1,
    )
    params = dict(
        env=env,
        start=dt.datetime.now(),
        verbose=1,
        print_every=10,
        save_every=1_000,
        log_path='log.txt',
        population_size=2,
        selection_pct=0.5,
        starting_population_size=2,
        weights=weights,
        min_p=0.1,
    )

    # save_random_maps(env=env, weights=weights, start_n=1_000, save_n=10, save_dir='maps/random-starting-points')
    # quit()

    param_sets = {}
    for value in [0.0, 0.5, 1.0]:
        params['params'] = ParameterCollection(
            expansion_population_bias=Parameter(-0.6, exp_factor=1),
            reduction_population_bias=Parameter(0.6, exp_factor=1),
            expansion_distance_bias=Parameter(-0.2, exp_factor=1),
            reduction_distance_bias=Parameter(0.2, exp_factor=1),
            expansion_surrounding_bias=Parameter(0.1, exp_factor=1),
            reduction_surrounding_bias=Parameter(-0.1, exp_factor=1),
            mutation_size=RangeParameter(0.0, 1.0, exp_factor=1 ** (1 / 20_000)),
            mutation_layers=RangeParameter(1, 1, exp_factor=1 ** (1 / 20_000), min_value=1),
            mutation_n=RangeParameter(1 / 17, 1 / 17, exp_factor=1 ** (1 / 20_000), max_value=34),
        )
        param_sets[value] = copy.deepcopy(params)

    test = AlgorithmTest(algorithm=GeneticRedistrictingAlgorithm, param_sets=param_sets, test_size=10,
                         start_map_path='maps/random-starting-points/2024-01-19-12-02-39')
    test.run(generations=300, results_path='test_results.txt')


if __name__ == '__main__':
    main()
