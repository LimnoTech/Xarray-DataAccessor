"""Defines custom data stacks of xarray data accessors."""
import abc
import dataclasses
from datetime import datetime, timedelta
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Optional,
)
from xarray_data_accessor.shared_types import (
    TimeInput,
    TableInput,
    ShapefileInput,
    RasterInput,
    CoordsTuple,
    BoundingBoxDict,
)


@dataclasses.dataclass(dict=True)
class DataStackInfo:
    """Stores all data within a data stack."""
    start_time: datetime
    end_time: datetime
    bounding_box: BoundingBoxDict
    time_resolution: timedelta
    space_resolution: Tuple[float, float]


class DataStackBase(abc.ABC):

    def __init__(
        self,
        start_time: TimeInput,
        end_time: TimeInput,
        coordinates: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[TableInput] = None,
        shapefile: Optional[ShapefileInput] = None,
        raster: Optional[RasterInput] = None,
        multithread: bool = True,
        use_dask: Optional[bool] = None,
    ) -> None:
        """Base class for all data stacks."""

        raise NotImplementedError

    @abc.abstractproperty
    def get_info(self) -> DataStackInfo:
        """Returns the data stack info."""
        raise NotImplementedError

    @abc.abstractmethod
    def _align_time(self) -> None:
        """Aligns the time dimension of all data sources."""
        raise NotImplementedError

    @abc.abstractmethod
    def _align_coords(self) -> None:
        """Aligns the coordinates of all data sources."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_stack(self) -> None:
        """Builds the data stack."""
        raise NotImplementedError
