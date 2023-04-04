import abc
from typing import (
    List,
    Dict,
    Union,
    Number,
)
from datetime import datetime
from xarray import Dataset
from xarray_data_accessor.shared_types import BoundingBoxDict


class DataAccessorBase(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abc.abstractproperty
    def supported_datasets(self) -> List[str]:
        """Returns all datasets that can be accessed."""""
        raise NotImplementedError

    @abc.abstractproperty
    def dataset_variables(self) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        raise NotImplementedError

    @abc.abstractmethod
    def _write_attrs(
        self,
        dataset_name: str,
        **kwargs,
    ) -> Dict[str, Union[str, Number]]:
        """Used to write aligned attributes to all sub datasets before merging"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(
        self,
        dataset: str,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
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
