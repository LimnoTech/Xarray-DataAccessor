"""Stores concrete implementations of base.ConverterBase.

Each python file stores a separate converter class that can be used convert a
xarray dataset to some other format.

Each DataAccessor is registered with the DataConverterFactory.
"""
# import all the data converters (this updates the factory class)
from xarray_data_accessor.data_converters.to_tables import ConvertToTable
from xarray_data_accessor.data_converters.to_gssha import ConvertToGSSHA

# init the wrapper class (updates class attributes)
from xarray_data_accessor.data_converters.factory import (
    DataConversionFunctions,
)
DataConversionFunctions.add_functions()
