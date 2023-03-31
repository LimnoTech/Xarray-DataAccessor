from xarray_data_accessor import (
    data_accessor,
)
from xarray_data_accessor.data_accessor import (
    DataAccessor,
)

# optional
try:
    from xarray_data_accessor import (
        era5_data_accessor,
        era5_datasets_info,
    )
except ImportError:
    pass
