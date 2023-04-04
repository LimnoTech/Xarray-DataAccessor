"""Stores concrete implementations of data_accessor_base.DataAccessorBase.

Each python file stores a separate DataAccessor that can be used to pull in
data from a specific source. In some cases, multiple datasets can be pulled
from the same source/DataAccessor.

Each DataAccessor is registered with the DataAccessorFactory.
"""
from xarray_data_accessor.data_accessors.era5_from_aws import AWSDataAccessor
from xarray_data_accessor.data_accessors.era5_from_cds import CDSDataAccessor
from xarray_data_accessor.data_accessors.data_accessor_base import DataAccessorBase
