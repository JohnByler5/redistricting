import datetime as dt
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

from .maps import DistrictMap


def is_number(x):
    return isinstance(x, (Number, np.number))


def since_start(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


class Parameter:
    """Class that allows ease of use for many useful functions in the algorithms, such as changing the parameter
     values over time and limiting those to certain minimum and maximum values."""

    def __init__(self, start_value, exp_factor=1.0, min_value=-np.inf, max_value=np.inf):
        assert is_number(start_value)
        assert is_number(exp_factor) and exp_factor > 0
        assert is_number(min_value) and min_value <= start_value
        assert is_number(max_value) and max_value >= start_value
        self.start_value = self.value = start_value
        self.exp_factor = exp_factor
        self.min_value = min_value
        self.max_value = max_value

    def tick(self):
        """Perform one time step for the parameter value, changing the value by an exponential factor."""
        self.value = np.clip(self.value ** self.exp_factor, self.min_value, self.max_value)


class RangeParameter:
    """Similar to the parameter class but uses a range of two values from which the algorithms can randomly sample."""

    def __init__(self, *values, exp_factor=1, min_value=-np.inf, max_value=np.inf):
        if not isinstance(exp_factor, tuple):
            exp_factor = (exp_factor, exp_factor)
        if not isinstance(min_value, tuple):
            min_value = (min_value, min_value)
        if not isinstance(max_value, tuple):
            max_value = (max_value, max_value)

        self.low, self.high = self.values = [
            Parameter(*args) for args in zip(values, exp_factor, min_value, max_value)
        ]
        self.start_values = values
        self.exp_factor = exp_factor
        self.min_value = min_value
        self.max_value = max_value

    def __iter__(self):
        return iter(self.values)

    def tick(self):
        self.low.tick()
        self.high.tick()

    def random(self):
        """Sample a random floating point value from the parameter range."""
        return np.random.random() * (self.high.value - self.low.value) + self.low.value

    def randint(self, scale=1, min_value=1):
        """Sample a random integer value from the parameter range."""
        return np.random.randint(
            max(int(self.low.value * scale), min_value),
            max(int(self.high.value * scale), min_value) + 1,
        )


class DictCollection(dict):
    """Provides ease of use functionality for working with dictionaries that are a collection of items. Used in the
    algorithm to easily work with metric weights."""

    def __init__(self, **items):
        super().__init__(items)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f'Item does not exist: {item}')

    def set_defaults(self, **defaults):
        for key, value in defaults.items():
            self.setdefault(key, value)


class ParameterCollection(DictCollection):
    """Builds off the DictCollection class but to work with parameters specifically."""

    def __init__(self, **params):
        super().__init__(**params)

    def tick(self):
        """Performs a time step for all parameters in the collection, mutating them all at once."""
        for param in self.values():
            param.tick()


class Algorithm:
    """Base algorithm class that is built off by the Genetic and Search algorithms. Provides common functionality."""

    def __init__(
            self,
            env,
            log_path=None,
            verbose=1,
            save_every=0,
            weights=None,
            params=None,
    ):
        self.start = dt.datetime.now()

        self.env = env

        self.verbose = verbose
        self.log_path = log_path
        open(self.log_path, 'w').close()

        self.save_every = save_every
        self.save_data_path = f'{env.save_data_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
        self.save_img_path = f'{env.save_img_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}.png'

        if weights is None:
            params = DictCollection()
        weights.set_defaults(
            contiguity=0,
            population_balance=1,
            compactness=1,
            win_margin=-1,
            efficiency_gap=-1,
        )
        self.weights = weights

        if params is None:
            params = ParameterCollection()
        self.params = params
        self.time_step_count = 0

        self._start_map = None

        # Load current map
        district_map = DistrictMap.load(env.current_data_path, env=self.env)
        self.current_fitness, self.current_metrics = district_map.calculate_fitness(DictCollection(
            contiguity=0,
            population_balance=-5,
            compactness=1,
            win_margin=-1,
            efficiency_gap=-1,
        ))

    def log(self, message, verbose=None):
        """Logs a message in a file and in the console (if indicated by verbose) with a timestamp."""
        message = f'{since_start(self.start)} - {message}'
        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                f.write(f'{message}\n')
        if verbose is not None and verbose <= self.verbose:
            print(message)

    def __enter__(self):
        if self.env.live_plot:
            plt.ion()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.env.live_plot:
            plt.ioff()
            plt.show()

    def set_start_map(self, map_):
        map_.env = self.env
        self._start_map = map_

    def run(self, generations): ...

    def _tick(self, district_map, metrics):
        """Performs a time step (or generation) for the algorithm."""
        self._plot(district_map)
        self.params.tick()
        self.time_step_count += 1

        elapsed = (dt.datetime.now() - self.start).total_seconds()
        return {
            "timeElapsed": f'{elapsed // 3600:02.0f}:{(elapsed % 3600) // 60:02.0f}:{elapsed % 60:02.0f}',
            "generation": self.time_step_count,
            "currentMap": {
                "imageUrl": self.env.current_img_path,
                "stats": {
                    "fitness": self.current_metrics['fitness'],
                    "pop-balance": self.current_metrics['population_balance'],
                    "win-margin": self.current_metrics['win_margin'],
                    "contiguity": self.current_metrics['contiguity'],
                    "compactness": self.current_metrics['compactness'],
                    "efficiency-gap": self.current_metrics['efficiency_gap'],
                }
            },
            "solutionMap": {
                "imageUrl": self.save_img_path,
                "stats": {
                    "fitness": metrics['fitness'],
                    "pop-balance": metrics['population_balance'],
                    "win-margin": metrics['win_margin'],
                    "contiguity": metrics['contiguity'],
                    "compactness": metrics['compactness'],
                    "efficiency-gap": metrics['efficiency_gap'],
                }
            }
        }

    def _plot(self, district_map):
        """Plots and saves the current district map if wanted and indicated for this timestep by the relevant
        parameters, i.e. 'save_every' and the save directory parameters."""
        if self.env.save_data_dir is not None and ((self.time_step_count % self.save_every) == 0):
            district_map.save(path=self.save_data_path)

        save_img = self.env.save_img_dir is not None and ((self.time_step_count % self.save_every) == 0)
        district_map.plot(save=save_img, save_path=self.save_img_path)
