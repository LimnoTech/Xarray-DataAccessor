import abc
from typing import (
    List,
    Dict,
    Union,
    TypedDict,
)
from numbers import Number
from datetime import datetime
from xarray import Dataset
#from xarray_data_accessor.shared_types import BoundingBoxDict


class AttrsDict(TypedDict):
    dataset_name: str
    institution: str
    x_dim: str
    y_dim: str
    EPSG: int
    time_step: str


class DataAccessorBase(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def supported_datasets(self) -> List[str]:
        """Returns all datasets that can be accessed."""""
        raise NotImplementedError

    @abc.abstractclassmethod
    def dataset_variables(self) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        raise NotImplementedError

    @abc.abstractproperty
    def attrs_dict(self) -> AttrsDict:
        """Used to write aligned attributes to all sub datasets before merging"""
        raise NotImplementedError

    @abc.abstractmethod
    def _parse_kwargs(self, kwargs_dict: TypedDict) -> None:
        """Parses kwargs and sets class attributes."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(
        self,
        dataset: str,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: dict,  # BoundingBoxDict,
        timezone: str,
        **kwargs,
    ) -> Dataset:
        """Gathers the desired variables for ones time/space AOI.

        Arguments:
            :param dataset: Dataset to access.
            :param variables: List of variables to access.
            :param start_dt: Datetime to start at (inclusive),
            :param end_dt: Datetime to stop at (exclusive).
            :param bbox: Dictionary with bounding box EPSG 4326 lat/longs.
            :param timezone: String specifying the desired timezone (see pytz docs).

        Returns:
            :return: xarray Dataset with the desired variables.
        """
        raise NotImplementedError
