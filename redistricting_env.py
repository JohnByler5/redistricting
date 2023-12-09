import datetime as dt

import geopandas as gpd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import Polygon
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, data_path, action_polygon_points=4, verbose=True, start=None, weights=None):
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
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_polygon_points * 2,), dtype=np.float32)
        self.n_features = self.n_districts * action_polygon_points * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.district_coords = []

        self.action_polygon_points = action_polygon_points

        self.weights = {
            'allocation': 0,
            'pop_balance': 1,
            'compactness': 0,
            'inter_pop_deviation': 0,
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

    def calculate_allocated(self, district_map):
        return 1 - district_map.n_unallocated / len(self.data)

    def calculate_population_balance(self, district_map):
        total_difference = sum(abs(district['population'] - self.ideal_pop)
                               for _, district in district_map.iterrows())
        max_difference = self.total_pop * (self.n_districts - 1) / self.n_districts * 2
        pop_balance = 1 - total_difference / max_difference
        return pop_balance

    def calculate_district_population_deviation(self, district_coords):
        district_geometry = Polygon(district_coords)
        assignments = np.array([district_geometry.contains(vtd.centroid) for vtd in self.data.geometry])
        district_vtds = self.data[assignments]
        pop_deviation = (district_vtds['population'].sum() - self.ideal_pop) / self.ideal_pop
        return pop_deviation

    @staticmethod
    def calculate_compactness(district_map):
        return (4 * np.pi * district_map.geometry.area / district_map.geometry.length ** 2).mean()

    def calculate_end_reward(self, district_map):
        allocated = self.calculate_allocated(district_map)
        pop_balance = self.calculate_population_balance(district_map)
        compactness = 0  # self.calculate_compactness(district_map)

        reward = allocated * self.weights['allocation'] + pop_balance * self.weights['pop_balance'] + \
            compactness * self.weights['compactness']
        metrics = {
            'Allocated': f'{allocated:.4%}',
            'Population Balance': f'{pop_balance:.4%}',
            'Compactness': f'{compactness:.4%}',
        }

        return reward, metrics

    def calculate_inter_reward(self, district_coords):
        pop_deviation = self.calculate_district_population_deviation(district_coords)

        reward = -abs(pop_deviation) * self.weights['inter_pop_deviation']
        metrics = {
            'Population Deviation': f'{pop_deviation:.4%}',
        }

        return reward, metrics

    @staticmethod
    def calculate_district_compactness(district_geometry):
        if district_geometry.is_empty:
            return 0
        area = district_geometry.area
        perimeter = district_geometry.length
        return 4 * np.pi * area / (perimeter ** 2)

    def construct_map(self):
        assignments, n_unallocated = [], 0
        district_geometries = [Polygon(coords) for coords in self.district_coords]
        for vtd in self.data.geometry:
            for district_index, district_geometry in enumerate(district_geometries):
                if district_geometry.contains(vtd.centroid):
                    assignments.append(district_index)
                    break
            else:
                distances = [vtd.centroid.distance(geometry.centroid) for geometry in district_geometries]
                assignments.append(np.argmin(distances))
                n_unallocated += 1
        assignments = np.array(assignments)

        district_data = []
        for district_index in range(self.n_districts):
            district_vtds = self.data[assignments == district_index]
            district_data.append({
                'district': district_index,
                'geometry': district_vtds.geometry.unary_union if not district_vtds.empty else
                Polygon(np.zeros(self.action_polygon_points * 2).reshape(self.action_polygon_points, 2)),
                'population': district_vtds['population'].sum()
            })

        districts_map = gpd.GeoDataFrame(district_data, crs=self.crs)
        districts_map.n_unallocated = n_unallocated
        return districts_map

    def step(self, action):
        district_coords = np.array(action).reshape((4, 2))
        min_x, min_y, max_x, max_y = self.data.total_bounds
        district_coords[:, 0] = min_x + (district_coords[:, 0] + 1) / 2 * (max_x - min_x)
        district_coords[:, 1] = min_y + (district_coords[:, 1] + 1) / 2 * (max_y - min_y)
        self.district_coords.append(district_coords)

        self.current_step += 1
        self.total_timesteps += 1
        if self.current_step == self.n_districts:
            reward, metrics = self.calculate_end_reward(self.construct_map())
            terminated, truncated = True, False

            self.total_simulations += 1
            if self.verbose:
                print(f'{time(self.start)} - Simulations: {self.total_simulations:,} | '
                      f'Timesteps: {self.total_timesteps:,} | Reward: {reward:.4f} | '
                      f'{" | ".join(f"{key}: {value}" for key, value in metrics.items())}')

        else:
            # reward, metrics = self.calculate_inter_reward(district_coords)
            # print(f'{time(self.start)} - Simulations: {self.total_simulations:,} | '
            #       f'Timesteps: {self.total_timesteps:,} | Reward: {reward:.4f} | '
            #       f'{" | ".join(f"{key}: {value}" for key, value in metrics.items())}')
            reward = 0
            terminated, truncated = False, False

        current_allocations = np.array(self.district_coords).flatten()
        observation = np.pad(current_allocations, (0, self.n_features - len(current_allocations)))
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.district_coords = []

        observation = np.zeros(self.n_features)

        reset_info = {}
        return observation, reset_info

    def render(self):
        pass

    def close(self):
        pass
