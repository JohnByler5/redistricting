import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import topojson as tp
from shapely import Polygon

from .union_cache import UnionCache


def get_utm_zone(longitude):
    """Gets the UTM zone from a longitude value. for purposes in the GeoDataFrame (inferring CRS)."""
    return int(((longitude + 180) / 6) % 60) + 1


def infer_utm_crs(data):
    """Infers the UTM CRS for the GeoDataFrame and converts the values as necessary. This effectively accounts for the
    z-value of a given area."""
    centroid = data.to_crs(epsg=4326).unary_union.centroid
    zone_number = get_utm_zone(centroid.x)
    hemisphere_prefix = 326 if centroid.y >= 0 else 327
    return f'EPSG:{hemisphere_prefix}{zone_number:02d}'


def refresh_data(data_path, simplified_path, simplify=False, save_neighbors=True):
    """Loads data and saves to a simplified parquet file for faster load and preprocessing files. Can simplify using
    topologically preserving algorithms if wanted to boost run times; however, contiguity performance can degrade
    and bugs can occur."""
    data = gpd.read_file(data_path)
    data.to_crs(infer_utm_crs(data), inplace=True)
    data.rename(columns={'TOTPOP20': 'population', 'PL10AA_TOT': 'population',
                         'PRES16D': 'democrat', 'EL16G_PR_D': 'democrat',
                         'PRES16R': 'republican', 'EL16G_PR_R': 'republican',
                         }, inplace=True)
    if simplify:
        data = tp.Topology(data, prequantize=False).toposimplify(50).to_gdf()
        data.geometry[~data.geometry.is_valid] = data.geometry[~data.geometry.is_valid].buffer(0)

    num_cols = data.select_dtypes(np.number).columns
    data[num_cols] = data[num_cols].astype(np.float64)
    data = data[['geometry', 'population', 'democrat', 'republican']]
    data.to_parquet(simplified_path)

    if save_neighbors:
        neighbors = gpd.sjoin(data, data, how='inner', predicate='touches')
        neighbors = neighbors[neighbors.index != neighbors['index_right']]
        neighbors.geometry = [Polygon([(0, 0), (0, 1), (1, 0)]) for _ in range(len(neighbors))]
        neighbors_path = f'{simplified_path.rstrip(".parquet")}-neighbors.parquet'
        neighbors.to_parquet(neighbors_path)


class RedistrictingEnv:
    """Environment class that contains the necessary data for the Voter Tabulation Districts (VTDs) of a state in a
    GeoDataFrame, like population, geometry, and vote counts, as well as other useful metadata such as the number of
    congressional districts for a given state. Also includes helper data such as a GeoDataFrame for all neighbors
    (touching) VTDs for every VTD and other functionality, such as a union cache to speed up union calculation times and
    live plotting functionality if wanted."""
    def __init__(self, state, n_districts, data_path, current_data_path, current_img_path, save_data_dir=None,
                 save_img_dir=None, live_plot=False):
        self.data = gpd.read_parquet(data_path)
        self.n_blocks = len(self.data)

        self.state = state
        assert isinstance(n_districts, int)
        assert n_districts > 0
        assert n_districts <= self.n_blocks

        neighbors_path = f'{data_path.rstrip(".parquet")}-neighbors.parquet'
        if os.path.exists(neighbors_path):
            self.neighbors = gpd.read_parquet(neighbors_path)
            self.neighbors.geometry = self.data.geometry[self.neighbors.index]
        else:
            self.neighbors = gpd.sjoin(self.data, self.data, how='inner', predicate='touches')
            self.neighbors = self.neighbors[self.neighbors.index != self.neighbors['index_right']]
        self.n_districts = n_districts
        self.ideal_population = self.data['population'].sum() / self.n_districts

        self.union_cache = UnionCache(geometries=self.data.geometry, size=1_000)

        self.current_data_path = current_data_path
        self.current_img_path = current_img_path
        self.save_data_dir = save_data_dir
        self.save_img_dir = save_img_dir
        self.live_plot = live_plot

        bounds = self.data.total_bounds
        aspect_ratio = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
        if not live_plot:
            matplotlib.use('agg')
        self.fig, self.ax = plt.subplots(figsize=(10, 10 / aspect_ratio))
