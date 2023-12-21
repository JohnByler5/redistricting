import copy

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection

from redistricting_env import RedistrictingEnv

UNALLOCATED = -1
DISTRICT_SUM_FEATURES = pd.Index(['population', 'republican', 'democrat'])
METRICS = ['contiguity', 'population_balance', 'compactness', 'win_margin', 'efficiency_gap']


def count_polygons(geometry):
    if isinstance(geometry, (Polygon, LineString)):
        return 1
    elif isinstance(geometry, (MultiPolygon, MultiLineString, GeometryCollection)):
        return len(geometry.geoms)
    else:
        raise ValueError(f'Incorrect geom type: "{type(geometry)}"')


class DistrictMap:
    def __init__(self, env, assignments=None, districts=None):
        assert isinstance(env, RedistrictingEnv)

        if assignments is None:
            self.assignments = np.full(env.n_blocks, UNALLOCATED)
        else:
            assert hasattr(assignments, '__iter__') or hasattr(assignments, '__getitem__')
            assert ((assignments >= -1) & (assignments < env.n_districts)).all()
            self.assignments = np.array(assignments)
            assert len(self.assignments) == env.n_blocks

        if districts is None:
            self.districts = gpd.GeoDataFrame(data=np.zeros((env.n_districts, len(DISTRICT_SUM_FEATURES))),
                                              geometry=np.full(env.n_districts, Polygon(np.zeros((3, 2)))),
                                              crs=env.data.crs)
            self.construct_districts()
        else:
            assert isinstance(districts, gpd.GeoDataFrame)
            assert DISTRICT_SUM_FEATURES.isin(districts.columns).all()
            assert (len(districts) % env.n_districts) == 0
            self.districts = districts

        self.env = env

    def copy(self):
        return DistrictMap(env=self.env, assignments=self.assignments.copy(), districts=copy.deepcopy(self.districts))

    def construct_districts(self):
        districts = self.env.data.groupby(by=self.assignments)
        self.districts[DISTRICT_SUM_FEATURES] = districts[DISTRICT_SUM_FEATURES].sum()
        self.districts.geometry = districts.index.apply(
            lambda x: self.env.union_cache.calculate_union(mask=self.assignments == x, geometries=districts[x])
        )

    def calculate_contiguity(self):
        total_breaks = sum(self.districts['geometry'].apply(count_polygons) - 1)
        max_breaks = len(self.env.data) - self.env.n_districts
        return 1 - total_breaks / max_breaks

    def calculate_population_balance(self):
        return (np.minimum(self.districts['population'], self.env.ideal_population) / np.maximum(
            self.districts['population'], self.env.ideal_population)).mean()

    def calculate_compactness(self):
        return (4 * np.pi * self.districts.geometry.area / self.districts.geometry.length ** 2).mean()

    def calculate_win_margin(self):
        return ((self.districts['democrat'] - self.districts['republican']).abs() / (
                self.districts['democrat'] + self.districts['republican'])).mean()

    def calculate_efficiency_gap(self):
        dem, rep = self.districts['democrat'], self.districts['republican']
        total_votes = dem + rep
        party_victory = (np.clip(self.districts['democrat'] - self.districts['republican'], -1, 1) + 1) / 2
        necessary_votes = np.floor(dem + rep) + 1
        dem_wasted_votes = (dem - (necessary_votes * party_victory)).sum()
        rep_wasted_votes = (rep - (necessary_votes * (1 - party_victory))).sum()
        return np.abs(dem_wasted_votes - rep_wasted_votes) / total_votes.sum()

    def randomize(self):
        self.assignments = np.full(self.env.n_blocks, UNALLOCATED)
        n_allocated = 0
        for district, starting_point in enumerate(np.random.choice(self.env.n_blocks, self.env.n_districts,
                                                                   replace=False)):
            self.assignments[starting_point] = district
            n_allocated += 1

        skipped_districts = set()
        districts, neighbors = np.array(range(self.env.n_districts)), self.env.neighbors['index_right']
        while n_allocated < len(self.env.data) and len(skipped_districts) != self.env.n_districts:
            np.random.shuffle(districts)
            for district in districts:
                if district in skipped_districts:
                    continue

                touching_vtds = neighbors[(self.assignments[neighbors.index] == district) &
                                          (self.assignments[neighbors] == UNALLOCATED)].unique()
                if not len(touching_vtds):
                    skipped_districts.add(district)
                    continue

                self.assignments[touching_vtds] = district
                n_allocated += len(touching_vtds)

        while n_allocated < len(self.env.data):
            for i in np.where(self.assignments == UNALLOCATED)[0]:
                unallocated = self.env.data.geometry[i]
                closest_vtd = np.argmin(unallocated.distance(self.env.data.geometry[self.assignments != UNALLOCATED]))
                if self.assignments[closest_vtd] == UNALLOCATED:
                    continue
                self.assignments[i] = self.assignments[closest_vtd]
                n_allocated += 1

        self.construct_districts()

    def plot(self, save_path, save=True):
        draw = self.env.live_plot and ('_temp_district' not in self.env.data or
                                       (self.env.data['_temp_district'] != self.assignments).any())
        if not (save or draw):
            return
        self.env.data['_temp_district'] = self.assignments
        self.env.ax.clear()
        self.env.data.plot(column='_temp_district', ax=self.env.ax, cmap='tab20')
        if draw:
            plt.draw()
            plt.pause(0.10)
        if save:
            self.env.fig.savefig(save_path)

    def set(self, index, to_district):
        from_district = self.assignments[index]
        self.assignments[index] = to_district

        sum_feature_differences = self.env.data[DISTRICT_SUM_FEATURES].iloc[index].values
        self.districts.loc[from_district, DISTRICT_SUM_FEATURES] = self.districts[
            DISTRICT_SUM_FEATURES].iloc[from_district].values - sum_feature_differences
        self.districts.loc[to_district, DISTRICT_SUM_FEATURES] = self.districts[
            DISTRICT_SUM_FEATURES].iloc[to_district].values + sum_feature_differences

        geometries = self.env.data.geometry[index].reset_index()
        self.districts.geometry.iloc[from_district] = self.districts.geometry.iloc[
            from_district].reset_index().difference(geometries).values
        self.districts.geometry.iloc[to_district] = self.districts.geometry.iloc[
            to_district].reset_index().union(geometries).values

    def get_border(self, district, reverse=False):
        if reverse:
            i1, i2 = self.env.neighbors.index, self.env.neighbors['index_right']
        else:
            i2, i1 = self.env.neighbors.index, self.env.neighbors['index_right']
        return self.env.neighbors[(self.assignments[i1] == district) &
                                  (self.assignments[i2] != district)]['index_right']

    def get_borders(self):
        return self.env.neighbors[(self.assignments[self.env.neighbors.index] != self.assignments[
            self.env.neighbors['index_right']])]['index_right']

    def mask(self, district):
        return self.assignments == district

    def repeated(self, n):
        return DistrictMap(self.env, districts=pd.concat([self.districts for _ in range(n)], ignore_index=True))

    def _calculate_groupby_contiguity(self):
        self.districts['contiguity'] = 1 - sum(self.districts.geometry.apply(count_polygons) - 1) / (
            len(self.env.data) - self.env.n_districts)

    def _calculate_groupby_population_balance(self):
        self.districts['population_balance'] = np.minimum(self.districts['population'], self.env.ideal_population) / \
                                               np.maximum(self.districts['population'], self.env.ideal_population)

    def _calculate_groupby_compactness(self):
        self.districts['compactness'] = 4 * np.pi * self.districts.geometry.area / self.districts.geometry.length ** 2

    def _calculate_groupby_win_margin(self):
        self.districts['win_margin'] = (self.districts['democrat'] - self.districts['republican']).abs() / (
            self.districts['democrat'] + self.districts['republican'])

    def _calculate_groupby_efficiency_gap(self):
        self.districts['total_votes'] = self.districts['democrat'] + self.districts['republican']
        votes_needed = np.floor(self.districts['total_votes']) + 1
        self.districts['rep_wv'] = self.districts['republican'] - (
            votes_needed * (self.districts['republican'] >= self.districts['democrat']))
        self.districts['dem_wv'] = self.districts['democrat'] - (
            votes_needed * (self.districts['democrat'] >= self.districts['republican']))

    def calculate_groupby_metrics(self):
        for metric in METRICS:
            getattr(self.districts, f'_calculate_groupby_{metric}')
        group_by = self.districts.groupby(by=self.districts.index // self.env.n_districts)
        outcomes = {}
        for metric in METRICS:
            if metric == 'efficiency_gap':
                outcome = np.abs(group_by['dem_wv'].sum() - group_by['rep_wv'].sum()) / group_by['total_votes'].sum()
            else:
                outcome = group_by[metric].mean()
            outcomes[metric] = outcome
        return outcomes


class DistrictMapCollection:
    def __init__(self, env, max_size=None, district_maps=None):
        assert isinstance(env, RedistrictingEnv)
        assert max_size is not None or district_maps is not None

        if district_maps is None:
            assert isinstance(max_size, int)
            assert max_size > 0
            self.size = 0
            self.max_size = max_size
            self.district_maps = np.array([DistrictMap(env=env) for _ in range(max_size)])
        else:
            assert hasattr(district_maps, '__iter__') or hasattr(district_maps, '__getitem__')
            assert all(isinstance(x, DistrictMap) for x in district_maps)
            assert all(district_map.env is env for district_map in district_maps)
            self.size = len(district_maps)
            if max_size is None:
                max_size = self.size
            self.district_maps = np.array([DistrictMap(env=env) for _ in range(max_size)])
            self.district_maps[:self.size] = np.array(district_maps)

        self.env = env

    def __iter__(self):
        return iter(self.district_maps)

    def __getitem__(self, item):
        return self.district_maps[item]

    def __setitem__(self, key, value):
        self.district_maps[key] = value

    def randomize(self):
        for i in range(self.size):
            self.district_maps[i].randomize()

    def select(self, indices):
        return DistrictMapCollection(self.env, max_size=self.max_size, district_maps=self.district_maps[indices])

    def add(self, district_map):
        assert self.size < self.max_size
        self.district_maps[self.size] = district_map
        self.size += 1
