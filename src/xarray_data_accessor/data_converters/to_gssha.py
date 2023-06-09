import xarray as xr
from xarray_data_accessor.data_converters.base import (
    DataConverterBase,
)
from typing import (
    Dict,
)


class ConvertToGSSHA(DataConverterBase):
    """Converts xarray datasets to GSSHA input files."""

    def get_precipitation(
        xarray_dataset: xr.Dataset,
        variable: str,
    ) -> str:
        ...

    def get_conversion_functions(
        cls,
    ) -> Dict[str, DataConverterBase.ConversionFunctionType]:
        """Returns a dictionary of conversion functions."""
        return {
            'get_precipitation_input': cls.get_precipitation,
        }
