import datetime as dt
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np


def is_number(x):
    return isinstance(x, (Number, np.number))


def since_start(start):
    return f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))}'


class Parameter:
    def __init__(self, start_value, exp_factor=1, min_value=-np.inf, max_value=np.inf):
        assert is_number(start_value)
        assert is_number(exp_factor) and exp_factor > 0
        assert is_number(min_value) and min_value <= start_value
        assert is_number(max_value) and max_value >= start_value
        self.start_value = self.value = start_value
        self.exp_factor = exp_factor
        self.min_value = min_value
        self.max_value = max_value

    def tick(self):
        self.value = np.clip(self.value ** self.exp_factor, self.min_value, self.max_value)


class RangeParameter:
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
        return np.random.random() * (self.high.value - self.low.value) + self.low.value

    def randint(self, scale=1, min_value=1):
        return np.random.randint(
            max(int(self.low.value * scale), min_value),
            max(int(self.high.value * scale), min_value) + 1,
        )


class DictCollection(dict):
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
    def __init__(self, **params):
        super().__init__(**params)

    def tick(self):
        for param in self.values():
            param.tick()


class Algorithm:
    def __init__(
            self,
            env,
            start=dt.datetime.now(),
            log_path='log.txt',
            verbose=1,
            save_every=0,
            weights=None,
            params=None,
    ):
        self.env = env

        self.start = start
        self.verbose = verbose
        self.log_path = log_path
        open(self.log_path, 'w').close()

        self.save_every = save_every

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

    def log(self, message, verbose=None):
        message = f'{since_start(self.start)} - {message}'
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

    def run(self, generations) -> [int]: ...

    def _tick(self, district_map):
        self._plot(district_map)
        self.params.tick()
        self.time_step_count += 1

    def _plot(self, district_map):
        if self.env.save_data_dir is not None and ((self.time_step_count % self.save_every) == 0):
            district_map.save(path=f'{self.env.save_data_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}.pkl')

        save_img = self.env.save_img_dir is not None and ((self.time_step_count % self.save_every) == 0)
        save_path = f'{self.env.save_img_dir}/{self.start.strftime("%Y-%m-%d-%H-%M-%S")}'
        district_map.plot(save=save_img, save_path=save_path)
