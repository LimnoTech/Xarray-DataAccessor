import abc
import xarray as xr
import typing


class DataConverterBase(abc.ABC):
    """Base class for data converters.

    NOTE: this class is not meant to be directly instantiated!
    """

    ConversionFunctionType = typing.Callable[[xr.Dataset], typing.Any]

    @abc.abstractclassmethod
    def get_conversion_functions(cls) -> typing.Dict[str, ConversionFunctionType]:
        """Returns a dictionary of conversion functions."""
        ...
