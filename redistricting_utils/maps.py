import copy
import datetime as dt
import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.errors
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union

from .env import RedistrictingEnv

UNALLOCATED = -1
DISTRICT_FEATURES = pd.Index(['population', 'republican', 'democrat'])
METRICS = ['contiguity', 'population_balance', 'compactness', 'win_margin', 'efficiency_gap']


def count_polygons(geometry):
    """Useful helper function to count the number of polygons in a geometry, used for contiguity calculations."""
    if isinstance(geometry, (Polygon, LineString)):
        return 1
    elif isinstance(geometry, (MultiPolygon, MultiLineString, GeometryCollection)):
        return len(geometry.geoms)
    else:
        raise ValueError(f'Incorrect geom type: "{type(geometry)}"')


def save_random_maps(env, save_dir, n=None, save_n=None, weights=None, balance_population=True,
                     balance_contiguity=True, start=dt.datetime.now()):
    """Generates random DistrictMap instances and saves them to a folder for use in an algorithm."""

    if n is None:
        n = 1_000_000
    if save_n is None:
        save_n = n
    assert n >= 1
    assert 1 <= save_n <= n
    assert save_n >= n or weights is not None

    dt_str = start.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(save_dir, dt_str)
    os.mkdir(save_dir)
    maps = []
    print(f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))} - Generating maps...')
    for i in range(n):
        district_map = DistrictMap(env=env)
        district_map.randomize(balance_population=balance_population, balance_contiguity=balance_contiguity)
        print(f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))} - ({i + 1}): '
              f'Contiguity: {district_map.calculate_contiguity():.4%} | '
              f'Population Balance: {district_map.calculate_population_balance():.4%} | '
              f'Compactness: {district_map.calculate_compactness():.4%} | '
              f'Win Margin: {district_map.calculate_win_margin():.4%} | '
              f'Efficiency Gap: {district_map.calculate_efficiency_gap():.4%}')

        if save_n >= n:
            save_path = f'{os.path.join(save_dir, str(i + 1))}.pkl'
            district_map.save(save_path)
        else:
            maps.append(district_map)

    if save_n < n:
        collection = DistrictMapCollection(env=env, max_size=len(maps), district_maps=maps)
        print(f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))} - '
              f'Calculating fitness scores...')
        fitness_scores, _ = collection.calculate_fitness(weights=weights)
        print(f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))} - '
              f'Selecting best {save_n:,} maps...')
        indices = sorted(fitness_scores, key=lambda x: fitness_scores[x], reverse=True)[:save_n]
        collection = collection.select(indices, new_max_size=save_n)
        print(f'{dt.timedelta(seconds=round((dt.datetime.now() - start).total_seconds()))} - '
              f'Saving {len(collection)} maps...')
        for i, map_ in enumerate(collection, 1):
            save_path = f'{os.path.join(save_dir, str(i))}.pkl'
            map_.save(save_path)


class DistrictMap:
    """Class that represents a given congressional district map solution for a RedistrictingEnv environment instance.
    Stores the assignments for each Voter Tabulation District (VTDs) as well as the aggregated geometry and metric dat
    for the current district assignments."""

    def __init__(self, env, assignments=None, districts=None):
        assert isinstance(env, RedistrictingEnv)
        self.env = env

        if assignments is None:
            self.assignments = np.full(env.n_blocks, UNALLOCATED)
        else:
            assert hasattr(assignments, '__iter__') or hasattr(assignments, '__getitem__')
            assert ((assignments >= -1) & (assignments < env.n_districts)).all()
            self.assignments = np.array(assignments)
            assert len(self.assignments) == env.n_blocks

        if districts is None:
            self.districts = gpd.GeoDataFrame(
                data=np.zeros((env.n_districts, len(DISTRICT_FEATURES))),
                columns=DISTRICT_FEATURES,
                geometry=np.full(env.n_districts, Polygon(np.zeros((3, 2)))),
                crs=env.data.crs
            )
            self.construct_districts()
        else:
            assert isinstance(districts, gpd.GeoDataFrame)
            assert DISTRICT_FEATURES.isin(districts.columns).all()
            assert (len(districts) % env.n_districts) == 0
            self.districts = districts

    def copy(self):
        return DistrictMap(env=self.env, assignments=self.assignments.copy(), districts=copy.deepcopy(self.districts))

    def save(self, path):
        """Saves DistrictMap instance (assignments only) to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self.assignments, f)

    @classmethod
    def load(cls, path, env):
        """Loads DistrictMap instance (assignments only) from a saved pickle file."""
        with open(path, 'rb') as f:
            return cls(env=env, assignments=pickle.load(f))

    def construct_districts(self):
        """Constructs the district geometries and aggregated data for the current assignments of VTDs."""
        allocated = self.assignments != UNALLOCATED
        if not allocated.sum():
            return
        districts = self.env.data[allocated].groupby(by=self.assignments[allocated])
        groups = pd.Index(districts.groups.keys())
        self.districts.loc[groups, DISTRICT_FEATURES] = districts[DISTRICT_FEATURES].sum()
        self.districts.loc[groups, 'geometry'] = districts.apply(
            lambda group: self.env.union_cache.calculate_union(mask=self.assignments == group.name, geometries=group)
        )

    def calculate_contiguity(self):
        """Calculates the contiguity metric. Real congressional district maps must be contiguous."""
        total_breaks = sum(self.districts['geometry'].apply(count_polygons) - 1)
        max_breaks = len(self.env.data) - self.env.n_districts
        return 1 - total_breaks / max_breaks

    def calculate_population_balance(self):
        """Calculates the population balance metric. Congressional districts should have equal or very close to equal
        populations."""
        return np.abs(self.districts.population - self.env.ideal_population).mean() / self.env.ideal_population

    def calculate_compactness(self):
        """Calculates the compactness metric. Uses the Polsby-Popper score to return a percentage of how close the
        district is to a circle (i.e. perfectly compact)."""
        return (4 * np.pi * self.districts.geometry.area / self.districts.geometry.length ** 2).mean()

    def calculate_win_margin(self):
        """Calculates the average win margin for the districts. More competitive districts (lower win margin) produce
        better congressional outcomes."""
        return ((self.districts['democrat'] - self.districts['republican']).abs() / (
                self.districts['democrat'] + self.districts['republican'])).mean()

    def calculate_efficiency_gap(self):
        """Calculates the efficiency gap metric, which is essentially a percentage of how Gerrymandered the maps is.
        Lower is better."""
        dem, rep = self.districts['democrat'], self.districts['republican']
        total_votes = dem + rep
        party_victory = (np.clip(self.districts['democrat'] - self.districts['republican'], -1, 1) + 1) / 2
        necessary_votes = np.floor(dem + rep) + 1
        dem_wasted_votes = (dem - (necessary_votes * party_victory)).sum()
        rep_wasted_votes = (rep - (necessary_votes * (1 - party_victory))).sum()
        return np.abs(dem_wasted_votes - rep_wasted_votes) / total_votes.sum()

    def randomize(self, balance_population=True, balance_contiguity=True):
        """Randomizes the district map assignments while ensuring contiguity and balancing population if desired."""

        self.assignments = np.full(self.env.n_blocks, UNALLOCATED)
        n_allocated = 0
        for district, starting_point in enumerate(np.random.choice(self.env.n_blocks, self.env.n_districts,
                                                                   replace=False)):
            self.assignments[starting_point] = district
            n_allocated += 1

        skipped_districts = set()
        populations = np.full(self.env.n_districts, 0)
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
                populations[district] += self.env.data.population[touching_vtds].sum()
                n_allocated += len(touching_vtds)

        while n_allocated < len(self.env.data):
            for i in np.where(self.assignments == UNALLOCATED)[0]:
                unallocated = self.env.data.geometry[i]
                closest_vtd = np.argmin(unallocated.distance(self.env.data.geometry[self.assignments != UNALLOCATED]))
                if self.assignments[closest_vtd] == UNALLOCATED:
                    continue

                district = self.assignments[closest_vtd]
                self.assignments[i] = district
                populations[district] += self.env.data.population[i]
                n_allocated += 1

        self.construct_districts()
        if balance_contiguity:
            self.balance_contiguity()
        if not balance_population:
            return
        self.balance_population()
        if balance_contiguity:
            self.balance_contiguity()

    def balance_contiguity(self):
        """Second layer of contiguity ensurance. Attempts to restore the map to be more contiguous where possible."""
        for district in range(self.env.n_districts):
            geometry = self.districts.geometry[district]
            if count_polygons(geometry) > 1:
                to_remove = list(geometry.geoms)
                to_remove.pop(np.argmax([polygon.area for polygon in to_remove]))
                for polygon in to_remove:
                    geometries = self.env.data.geometry[self.assignments == district]
                    vtds = geometries.index[polygon.contains(geometries.centroid)]
                    vtds = vtds[vtds.isin(self.env.neighbors.index)]
                    if not len(vtds):
                        continue
                    touching = self.assignments[self.env.neighbors['index_right'][vtds]]
                    touching = touching[touching != district]
                    if not len(touching):
                        continue
                    add_district = np.random.choice(touching)
                    self.set(vtds, add_district)

    def balance_population(self):
        """Balances the population of the district assignments to be very minimal. Usually produces a result within
        0.5% average deviation of the ideal population for each district, but takes a couple of minutes to run."""

        neighbors = self.env.neighbors['index_right']
        balance = np.abs(self.districts.population - self.env.ideal_population).mean() / self.env.ideal_population
        min_balance = balance
        same_prev_count, used, failed_count = 0, {}, 0
        while same_prev_count < 2_000 and balance >= 0.002:
            available = self.districts[[bool(i not in used or len(used[i])) for i in self.districts.index]]
            district = available.index[np.argmin(available.population)]
            if self.districts.population[district] > self.env.ideal_population:
                used = {}
                failed_count += 1
                if failed_count >= 10:
                    break
                continue

            touching_vtds = neighbors[self.assignments[neighbors.index] == district].unique()
            touching_vtds = touching_vtds[self.assignments[touching_vtds] != district]
            if not len(touching_vtds):
                used[district] = []
                continue
            used.setdefault(district, list(self.assignments[touching_vtds]))
            target = used[district].pop(np.random.randint(low=0, high=len(used[district])))
            touching_vtds = touching_vtds[self.assignments[touching_vtds] == target]

            max_population = self.env.ideal_population * (1 + balance / self.env.n_districts)
            if self.districts.population[district] + self.env.data.population[touching_vtds].sum() > max_population:
                new = self.env.data.geometry[touching_vtds].centroid
                to, from_ = self.districts.geometry[district].centroid, self.districts.geometry[target].centroid
                key = (new.x - to.x) ** 2 + (new.y - to.y) ** 2 - (new.x - from_.x) ** 2 - (new.y - from_.y) ** 2
                touching_vtds = touching_vtds[key.argsort()]

                bounds = np.array([0, len(touching_vtds)])
                while True:
                    selected = touching_vtds[:max(int(bounds.mean()), bounds[0] + 1)]
                    if bounds[0] + 1 >= bounds[1]:
                        break
                    which = int(self.districts.population[district] + self.env.data.population[touching_vtds].sum() >
                                self.env.ideal_population * (1 + balance / self.env.n_districts))
                    bounds[which] = len(selected)
                touching_vtds = selected

            if not len(touching_vtds):
                continue

            mask = self.assignments == target
            geometries = self.env.data.geometry[mask]
            union = self.env.union_cache.calculate_union(mask, geometries)
            if count_polygons(union) > count_polygons(self.districts.geometry[target]):
                new = self.env.union_cache.calculate_union(self.env.data.index.isin(touching_vtds))
                to_add = [polygon for polygon in union.geoms if polygon.touches(new)]
                if not to_add:
                    continue
                to_add.pop(np.argmax([polygon.area for polygon in to_add]))
                for polygon in to_add:
                    contains = geometries.index[polygon.contains(geometries.centroid)].to_numpy()
                    if len(contains):
                        touching_vtds = np.concatenate([touching_vtds, contains])

            self.set(touching_vtds, district)
            prev_min_balance = min_balance
            min_balance = min(min_balance, balance)
            balance = np.abs(self.districts.population - self.env.ideal_population).mean() / self.env.ideal_population
            if min_balance == prev_min_balance:
                same_prev_count += 1
            else:
                same_prev_count = 0
                used = {}
            failed_count = 0

    def plot(self, save_path=None, save=True):
        """Plots and saves (if desired) the current district map."""
        save = save and save_path is not None
        draw = self.env.live_plot and ('_temp_district' not in self.env.data or
                                       (self.env.data['_temp_district'] != self.assignments).any())
        if not (save or draw):
            return
        self.env.data['_temp_district'] = self.assignments
        self.env.ax.clear()
        self.env.data.plot(column='_temp_district', ax=self.env.ax, cmap='tab20')
        self.env.ax.axis('off')
        if draw:
            plt.draw()
            plt.pause(0.10)
        if save:
            self.env.fig.savefig(save_path, bbox_inches='tight')

    def set(self, index, to_district):
        """Modifies the assignments and aggregated district geometries and data to allocate certain VTDs to a different
        district(s)."""
        if not hasattr(index, '__iter__'):
            index = np.array([index])
        if not hasattr(to_district, '__iter__'):
            to_district = np.full(len(index), to_district)

        from_district = self.assignments[index]
        self.assignments[index] = to_district

        geometries = self.env.data.geometry[index]
        for assignments, weight, f in ((from_district, -1, 'difference'), (to_district, 1, 'union')):
            differences = self.env.data[DISTRICT_FEATURES].iloc[index].groupby(assignments).sum()
            unique = np.unique(assignments)
            self.districts.loc[unique, DISTRICT_FEATURES] += differences * weight

            f = getattr(self.districts.geometry.loc[unique].reset_index(drop=True), f)
            unions = geometries.groupby(assignments).apply(unary_union).reset_index(drop=True)
            unions.crs = self.env.data.crs
            try:
                self.districts.loc[unique, 'geometry'] = f(unions).values
            except shapely.errors.GEOSException:
                self.construct_districts()
                break

    def get_border(self, district, reverse=False):
        """Gets all the bordering VTDs of a given district, inside or outside determined by 'reverse.'"""
        if reverse:
            i2, i1 = self.env.neighbors.index, self.env.neighbors['index_right']
        else:
            i1, i2 = self.env.neighbors.index, self.env.neighbors['index_right']
        return pd.Series(self.env.neighbors[(self.assignments[i1] == district) &
                                            (self.assignments[i2] != district)]['index_right'].unique())

    def get_borders(self):
        """Gets all the borders of all districts of the map for use in the algorithm."""
        return self.env.neighbors[(self.assignments[self.env.neighbors.index] != self.assignments[
            self.env.neighbors['index_right']])]['index_right']

    def mask(self, district):
        """Calculates a boolean mask of VTD selections for a given district."""
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

    def calculate_fitness(self, weights):
        score = 0
        metrics = {'fitness': '0'}
        for metric, weight in weights.items():
            result = getattr(self, f'calculate_{metric}')()
            score += result * weight
            metrics[metric] = f'{result:.4%}'
        metrics['fitness'] = f'{score:.4f}'
        return score, metrics


class DistrictMapCollection:
    """Helper class for a collection/population of DistrictMap instances."""

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
            self.max_size = max_size
            self.district_maps = np.array([DistrictMap(env=env) for _ in range(max_size)])
            self.district_maps[:self.size] = np.array(district_maps)

        self.env = env

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.district_maps[:self.size])

    def __getitem__(self, item):
        return self.district_maps[item]

    def __setitem__(self, key, value):
        self.district_maps[key] = value

    def is_full(self):
        assert self.size <= self.max_size
        return self.size == self.max_size

    def empty_space(self):
        return self.max_size - self.size

    def randomize(self, *args, **kwargs):
        """Randomizes all district maps in the collection."""
        for i in range(self.max_size):
            self.district_maps[i].randomize(*args, **kwargs)
        self.size = self.max_size

    def fill_random(self, *args, **kwargs):
        """Fills the remaining space within the collection with random maps."""
        for i in range(self.size, self.max_size):
            self.district_maps[i].randomize(*args, **kwargs)
        self.size = self.max_size

    def select(self, indices, new_max_size=None):
        """Selects specific maps and returns a new collection of them."""
        if new_max_size is None:
            new_max_size = self.max_size
        return DistrictMapCollection(self.env, max_size=new_max_size, district_maps=self.district_maps[indices])

    def add(self, other):
        """Adds a new map to the collection."""
        assert isinstance(other, (DistrictMap, DistrictMapCollection))
        assert self.env is other.env

        if isinstance(other, DistrictMap):
            new_size = self.size + 1
        else:
            new_size = self.size + other.size
            other = other.district_maps[:other.size]
        assert new_size <= self.max_size

        self.district_maps[self.size:new_size] = other
        self.size = new_size

    def calculate_fitness(self, weights):
        """Calculates metrics and fitness from a set of metric weights for each of the district maps for use in the
        algorithms."""
        scores, metrics = {}, {}
        for i, district_map in enumerate(self):
            scores[i] = 0
            metrics[i] = {'fitness': '0'}
            for metric, weight in weights.items():
                result = getattr(district_map, f'calculate_{metric}')()
                scores[i] += result * weight
                metrics[i][metric] = f'{result:.4%}'
            metrics[i]['fitness'] = f'{scores[i]:.4f}'
        return scores, metrics
