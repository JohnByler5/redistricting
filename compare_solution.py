import geopandas as gpd

from algorithm import DictCollection
from maps import DistrictMap
from redistricting_env import RedistrictingEnv, infer_utm_crs


def calculate_fitness(district_map, weights):
    score = 0
    metrics = {'fitness': '0'}
    for metric, weight in weights.items():
        result = getattr(district_map, f'calculate_{metric}')()
        score += result * weight
        metrics[metric] = f'{result:.4%}'
    metrics['fitness'] = f'{score:.4f}'
    return score, metrics


def main():
    env = RedistrictingEnv('data/pa/simplified.parquet', n_districts=17, live_plot=False, save_img_dir='maps')
    districts = gpd.read_file('data/pa/current-boundaries.shp')
    solution = DistrictMap.load('maps/data/2024-01-20-00-05-33.pkl')
    districts.to_crs(infer_utm_crs(districts), inplace=True)
    centroids = gpd.GeoDataFrame(env.data, geometry=env.data.geometry.centroid)
    assignments = gpd.sjoin(centroids, districts, how='left', op='within')['index_right'].values
    district_map = DistrictMap(env=env, assignments=assignments)
    district_map.save('maps/current/pa')

    weights = DictCollection(
        contiguity=0,
        population_balance=5,
        compactness=1,
        win_margin=-1,
        efficiency_gap=-1,
    )

    score, metrics = calculate_fitness(district_map, weights)
    metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in metrics.items()]
    print(f'Current District Metrics: {" | ".join(metric_str for metric_str in metric_strs)}')

    score, metrics = calculate_fitness(solution, weights)
    metric_strs = [f'{" ".join(key.title().split("_"))}: {value}' for key, value in metrics.items()]
    print(f'Solution Metrics: {" | ".join(metric_str for metric_str in metric_strs)}')


if __name__ == '__main__':
    main()
