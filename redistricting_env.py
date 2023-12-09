import datetime as dt

import geopandas as gpd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Polygon, MultiPolygon


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


class RedistrictingEnv(gym.Env):
    def __init__(self, data_path, verbose=True, start=None, weights=None):
        super(RedistrictingEnv, self).__init__()
        self.data = gpd.read_file(data_path)
        self.crs = infer_utm_crs(self.data)
        self.data = self.data.to_crs(self.crs)
        self.data.rename(columns={'P0010001': 'population'}, inplace=True)

        self.total_pop = self.data['population'].sum()
        self.n_districts = 17
        self.ideal_pop = self.total_pop / self.n_districts
        self.train_data = self.preprocess_data()

        self.current_step = 0
        self.action_space = spaces.MultiDiscrete([self.n_districts] * len(self.data))
        self.n_features = len(self.data)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)

        self.weights = {
            'contiguity': 1,
            'pop_balance': 0,
            'compactness': 0,
        }
        for key in weights:
            if key in self.weights:
                self.weights[key] = weights[key]

        self.total_simulations = 0
        self.total_timesteps = 0
        self.verbose = verbose
        if start is None:
            start = dt.datetime.now()
        self.start = start

    def preprocess_data(self):
        self.data['centroid_x'] = self.data.geometry.centroid.x
        self.data['centroid_y'] = self.data.geometry.centroid.y
        self.data['area'] = self.data.geometry.area
        self.data['perimeter'] = self.data.geometry.length
        self.data['aspect_ratio'] = self.data.geometry.bounds.apply(
            lambda x: (x.maxx - x.minx) / (x.maxy - x.miny), axis=1)
        self.data['area_to_bbox_ratio'] = self.data.area / self.data.geometry.envelope.area

        self.data['population_pct'] = self.data['population'] / self.ideal_pop

        scaler = StandardScaler()
        features = ['centroid_x', 'centroid_y', 'area', 'perimeter', 'aspect_ratio', 'area_to_bbox_ratio',
                    'population_pct']
        train_data = scaler.fit_transform(self.data[features])

        return train_data

    def calculate_contiguity(self, district_map):
        total_breaks = sum(count_polygons(district['geometry']) - 1 for _, district in district_map.iterrows())
        max_breaks = len(self.data) - self.n_districts
        return 1 - total_breaks / max_breaks

    def calculate_population_balance(self, district_map):
        total_difference = sum(abs(district['population'] - self.ideal_pop)
                               for _, district in district_map.iterrows())
        max_difference = self.total_pop * (self.n_districts - 1) / self.n_districts * 2
        return 1 - total_difference / max_difference

    @staticmethod
    def calculate_compactness(district_map):
        return (4 * np.pi * district_map.geometry.area / district_map.geometry.length ** 2).mean()

    def calculate_reward(self, district_map):
        contiguity = self.calculate_contiguity(district_map)
        pop_balance = 0  # self.calculate_population_balance(district_map)
        compactness = 0  # self.calculate_compactness(district_map)

        reward = contiguity * self.weights['contiguity'] + pop_balance * self.weights['pop_balance'] + \
            compactness * self.weights['compactness']
        metrics = {
            'Contiguity': f'{contiguity:.4%}',
            'Population Balance': f'{pop_balance:.4%}',
            'Compactness': f'{compactness:.4%}',
        }

        return reward, metrics

    def construct_map(self, assignments):
        district_data = []
        for district_index in range(self.n_districts):
            district_vtds = self.data[assignments == district_index]
            district_data.append({
                'district': district_index,
                'geometry': district_vtds.geometry.unary_union
                if not district_vtds.empty else Polygon(np.zeros(6).reshape(3, 2)),
                'population': district_vtds['population'].sum()
            })

        return gpd.GeoDataFrame(district_data, crs=self.crs)

    def step(self, action):
        self.current_step += 1
        self.total_timesteps += 1

        reward, metrics = self.calculate_reward(self.construct_map(action))
        terminated, truncated = True, False

        self.total_simulations += 1
        if self.verbose:
            print(f'{time(self.start)} - Simulations: {self.total_simulations:,} | '
                  f'Timesteps: {self.total_timesteps:,} | Reward: {reward:.4f} | '
                  f'{" | ".join(f"{key}: {value}" for key, value in metrics.items())}')

        observation = self.data['population_pct'].values
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        observation = self.data['population_pct'].values
        reset_info = {}

        return observation, reset_info

    def render(self):
        pass

    def close(self):
        pass
