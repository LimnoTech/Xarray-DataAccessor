import abc
import xarray as xr
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    TypedDict,
)
from datetime import datetime
from rasterio.crs import CRS
from geopandas import GeoDataFrame
from pathlib import Path
from xarray_data_accessor.shared_types import BoundingBoxDict

Shapefile = Union[str, Path, GeoDataFrame]
CoordsTuple = Tuple[float, float]  # lat/long
ResolutionTuple = Tuple[Union[int, float], Union[int, float]]


class BoundingBoxDict(TypedDict):
    west: float
    south: float
    east: float
    north: float


class ResampleDict(TypedDict):
    width: int
    height: int
    resampling_method: str
    crs: CRS
    index: int


class DataGrabberDict(TypedDict):
    point_data: xr.Dataset
    point_ids: List[str]
    xy_dims: Tuple[str, str]


@abc.ABC
class DataAccessorBase:

    @abc.abstractmethod
    def __init__(
        self,
        dataset_name: str,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abc.abstractproperty
    def data_source_classes(self) -> Dict[str, object]:
        """Stores all sub data sources that can be used to access data.

        For example, ERA5 data can come from either AWS or CDS API.
        Alternatively one could make an ElevationDataAccessor class where
        param:dataset_name controls which sub source is used.

        Returns:
            :return: dict with data source names and objects used to access
                said source.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_attrs(self) -> None:
        """Used to write aligned attributes to all sub datasets before merging"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(
        self,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        **kwargs,
    ) -> xr.Dataset:
        """Gathers the desired variables for ones time/space AOI.

        Arguments:
            :param variables: List of variables to access.
            :param start_dt: Datetime to start at (inclusive),
            :param end_dt: Datetime to stop at (exclusive).
            :param bbox: Dictionary with bounding box EPSG 4326 lat/longs.

        Returns:
            :return: xarray Dataset with the desired variables.
        """
        raise NotImplementedError
