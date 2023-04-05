"""PyTest unit tests for core functionality of xarray_data_accessor package."""
import xarray_data_accessor
from xarray_data_accessor import (
    DataAccessorFactory,
)
from xarray_data_accessor.data_accessors.base import (
    DataAccessorBase,
)
from xarray_data_accessor.data_accessors.factory import (
    DataAccessorFactory,
)


def test_name_space():
    """Test that the xarray_data_accessor namespace is the correct length."""
    count = 0
    names = [
        'get_xarray_dataset',
        'get_bounding_box',
        'convert_timezone',
        'resample_dataset',
        'get_data_tables',
        'delete_temp_files',
    ]
    for name in names:
        if name in dir(xarray_data_accessor):
            count += 1
    assert count == len(names)


def test_data_accessors_dict():
    """Test that the DataAccessorFactory contains registered data accessors."""
    data_accessors = DataAccessorFactory.data_accessor_objects()
    assert isinstance(data_accessors, dict)
    assert len(data_accessors) > 0
    for data_accessor in data_accessors.values():
        assert issubclass(data_accessor, DataAccessorBase)


def test_datasets():
    """Test that there are datasets in the data accessors"""
    # get the supported datasets for the data accessor
    for data_accessor_name in DataAccessorFactory.data_accessor_names():
        supported_datasets = DataAccessorFactory.supported_datasets()[
            data_accessor_name]
        assert isinstance(supported_datasets, list)
        assert len(supported_datasets) > 0


def test_dataset_variables():
    """Test that there are supported variables list for each dataset"""
    # get the supported datasets for the data accessor
    for data_accessor_name in DataAccessorFactory.data_accessor_names():
        datasets = DataAccessorFactory.supported_datasets()[
            data_accessor_name]

        # get the supported variables for each dataset
        for dataset_name in datasets:
            supported_variables = DataAccessorFactory.supported_variables(
                data_accessor_name,
                dataset_name,
            )
            assert isinstance(supported_variables, list)
            assert len(supported_variables) > 0
