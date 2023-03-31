from xarray_data_accessor.data_accessors.data_accessor_base import DataAccessorBase


class DataAccessorProduct:
    def __init__(
        self,
        args,
    ) -> None:
        DataAccessorFactory.register(args)
        self._args = args


class DataAccessorFactory:

    __data_accessors = {}

    @classmethod
    def register(
        cls,
        data_accessor: DataAccessorBase
    ):
        cls.__data_accessors[data_accessor.__name__] = data_accessor

    @classmethod
    def get_data_accessor(
        cls,
        data_accessor_name: str,
        *args,
        **kwargs,
    ) -> DataAccessorBase:
        return cls.__data_accessors[data_accessor_name](*args, **kwargs)
