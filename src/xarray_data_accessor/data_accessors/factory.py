from xarray_data_accessor.data_accessors.base import DataAccessorBase
from typing import (
    Dict,
    List,
)


class DataAccessorProduct:
    def __init__(
        cls,
        args,
    ) -> None:
        DataAccessorFactory.register(args)
        cls._args = args


class DataAccessorFactory:

    __data_accessors = {}
    __supported_datasets = {}

    @classmethod
    def register(
        cls,
        data_accessor: DataAccessorBase,
    ):
        cls.__data_accessors[data_accessor.__name__] = data_accessor

    @classmethod
    def data_accessor_objects(cls) -> Dict[str, DataAccessorBase]:
        return cls.__data_accessors

    @classmethod
    def data_accessor_names(cls) -> List[str]:
        return list(cls.__data_accessors.keys())

    @classmethod
    def supported_datasets(cls) -> Dict[str, List[str]]:
        if not cls.__supported_datasets:
            for name, da in cls.__data_accessors.items():
                cls.__supported_datasets[name] = da.supported_datasets()
        return cls.__supported_datasets

    @classmethod
    def supported_variables(
        cls,
        data_accessor_name: str,
        dataset_name: str,
    ) -> List[str]:
        return cls.__data_accessors[data_accessor_name].dataset_variables()[dataset_name]

    @classmethod
    def get_data_accessor(
        cls,
        data_accessor_name: str,
        *args,
        **kwargs,
    ) -> DataAccessorBase:
        return cls.__data_accessors[data_accessor_name](*args, **kwargs)
