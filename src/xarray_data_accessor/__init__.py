"""xarray_data_accessor package.

This package provides a set of functions for pulling in, and working with
gridded environmental data.
"""

__version__ = '0.1.0'

from xarray_data_accessor.core_functions import (
    get_xarray_dataset,
    get_bounding_box,
    convert_timezone,
    resample_dataset,
    get_data_tables,
    delete_temp_files,
)
from xarray_data_accessor.data_accessors.factory import (
    DataAccessorFactory,
)
