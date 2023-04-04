from xarray_data_accessor.data_accessors import DataAccessorBase
from typing import (
    Dict,
    List,
)


class DataAccessorProduct:
    def __init__(
        self,
        args,
    ) -> None:
        DataAccessorFactory.register(args)
        self._args = args


class DataAccessorFactory:

    __data_accessors = {}
    __supported_datasets = {}

    @classmethod
    def register(
        cls,
        data_accessor: DataAccessorBase
    ):
        cls.__data_accessors[data_accessor.__name__] = data_accessor

    @property
    def data_accessor_objects(self) -> Dict[str, DataAccessorBase]:
        return self.__data_accessors

    @property
    def data_accessor_names(self) -> List[str]:
        return list(self.__data_accessors.keys())

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        if not self.__supported_datasets:
            for name, data_accessor in self.__data_accessors.items():
                self.__supported_datasets[name] = data_accessor.supported_datasets
        return self.__supported_datasets

    @classmethod
    def supported_variables(
        cls,
        data_accessor_name: str,
        dataset_name: str,
    ) -> List[str]:
        return cls.__data_accessors[data_accessor_name].dataset_variables[dataset_name]

    @classmethod
    def get_data_accessor(
        cls,
        data_accessor_name: str,
        *args,
        **kwargs,
    ) -> DataAccessorBase:
        return cls.__data_accessors[data_accessor_name](*args, **kwargs)
