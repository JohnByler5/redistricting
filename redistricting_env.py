import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import topojson as tp

from union_cache import UnionCache


def get_utm_zone(longitude):
    return int(((longitude + 180) / 6) % 60) + 1


def infer_utm_crs(data):
    centroid = data.to_crs(epsg=4326).unary_union.centroid
    zone_number = get_utm_zone(centroid.x)
    hemisphere_prefix = 326 if centroid.y >= 0 else 327
    return f'EPSG:{hemisphere_prefix}{zone_number:02d}'


def refresh_data(data_path, simplified_path):
    data = gpd.read_file(data_path)
    data.to_crs(infer_utm_crs(data), inplace=True)
    data.rename(columns={'TOTPOP20': 'population', 'PL10AA_TOT': 'population',
                         'PRES16D': 'democrat', 'EL16G_PR_D': 'democrat',
                         'PRES16R': 'republican', 'EL16G_PR_R': 'republican',
                         }, inplace=True)
    # data = tp.Topology(data, prequantize=False).toposimplify(50).to_gdf()
    # data.geometry[~data.geometry.is_valid] = data.geometry[~data.geometry.is_valid].buffer(0)
    data.to_parquet(simplified_path)


class RedistrictingEnv:
    def __init__(self, data_path, n_districts, live_plot=False, save_data_dir=None, save_img_dir=None):
        self.data = gpd.read_parquet(data_path)
        num_cols = self.data.select_dtypes(np.number).columns
        self.data[num_cols] = self.data[num_cols].astype(np.float64)
        self.n_blocks = len(self.data)

        assert isinstance(n_districts, int)
        assert n_districts > 0
        assert n_districts <= self.n_blocks

        self.neighbors = gpd.sjoin(self.data, self.data, how='inner', predicate='touches')
        self.neighbors = self.neighbors[self.neighbors.index != self.neighbors['index_right']]

        self.n_districts = n_districts
        self.ideal_population = self.data['population'].sum() / self.n_districts

        self.union_cache = UnionCache(geometries=self.data.geometry, size=1_000)

        self.live_plot = live_plot
        self.save_data_dir = save_data_dir
        self.save_img_dir = save_img_dir
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
