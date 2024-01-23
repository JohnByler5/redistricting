import geopandas as gpd
import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import Polygon


class UnionCache:
    """Helper class built to provide significant speed improvements to the process of combining multiple geometries,
    a.k.a "union." It stores previous union calculations to compare with what is needed for the new calculation, and
    tries to find the previous stored union that most closely matches the new. This will then make the amount of changes
    much less, as all that needs done is small amounts of adding and subtracting of the slight mismatching geometries.
    Speeds up the algorithm by 2x-3x."""

    def __init__(self, geometries, size=1_000):
        assert isinstance(geometries, gpd.GeoSeries)
        assert isinstance(size, int)
        assert size >= 0
        self.geometries = geometries
        self.size = size
        self._cache = gpd.GeoDataFrame(data=[[[None for _ in range(len(geometries))]] for _ in range(size)],
                                       columns=['mask'], geometry=[None for _ in range(size)])
        self._index = 0
        self._count = 0

    def calculate_union(self, mask, geometries=None):
        """Calculates a union from a mask of a series of geometries telling which geometries need combined."""
        if not mask.any():
            return Polygon(np.zeros((3, 2)))
        if geometries is None:
            geometries = self.geometries[mask]

        union, prev_mask, score = self._find_closest_union(geometries.total_bounds)
        if score > 0.2:
            try:
                to_subtract = self.geometries[prev_mask & (~mask)]
                if not to_subtract.empty:
                    union = union.difference(to_subtract.unary_union)
                to_add = self.geometries[mask & (~prev_mask)]
                if not to_add.empty:
                    union = union.union(to_add.unary_union)
            except GEOSException:
                union = None
        else:
            union = None

        if union is None:
            union = geometries.geometry.unary_union

        self._add(mask, union)
        return union

    def _find_closest_union(self, bounds):
        """Finds the closest stored union to what needs calculated from the bounds of the geometries to calculate."""
        scores = self._calculate_bbox_score(bounds)
        if not len(scores):
            return None, None, 0
        i = scores.argmax()
        return self._cache.geometry[i], self._cache['mask'][i], scores[i]

    def _calculate_bbox_score(self, bounds):
        """Calculates the similarity scores for each of the stores unions to the given bounds set for the new
        calculation."""

        bounds_series = self._cache.geometry[:self._count].bounds
        u_minx, u_miny, u_maxx, u_maxy = [bounds_series[s] for s in ['minx', 'miny', 'maxx', 'maxy']]

        intersection_width = np.maximum(np.minimum(u_maxx, bounds[2]) - np.maximum(u_minx, bounds[0]), 0)
        intersection_height = np.maximum(np.minimum(u_maxy, bounds[3]) - np.maximum(u_miny, bounds[1]), 0)
        intersection_area = intersection_width * intersection_height

        u_area = (u_maxx - u_minx) * (u_maxy - u_miny)
        b_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        if b_area == 0:
            return np.zeroes(self._count)

        scores = (intersection_area * 2 - u_area) / b_area
        scores[u_area == 0] = 0
        return scores.values

    def _add(self, mask, union):
        """Adds the newly calculated union to the cache and removes an old one if needed."""
        self._cache.at[self._index, 'mask'] = mask
        self._cache.at[self._index, 'geometry'] = union
        self._index = (self._index + 1) % self.size
        self._count = min(self._count + 1, self.size)
