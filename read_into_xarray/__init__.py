from read_into_xarray import (
    data_accessor,
)
from read_into_xarray.data_accessor import (
    DataAccessor,
)

# optional
try:
    from read_into_xarray import (
        era5_data_accessor,
        era5_datasets_info,
    )
except ImportError:
    pass
