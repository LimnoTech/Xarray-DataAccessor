import logging
import warnings
import rioxarray
import xarray as xr
from typing import (
    Dict,
    TypedDict,
    Union,
    Any,
)
from numbers import Number
from xarray_data_accessor.data_accessors.base import DataAccessorBase


def apply_kwargs(
    accessor_object: DataAccessorBase,
    accessor_kwargs_dict: TypedDict,
    kwargs_dict: Dict[str, Any],
) -> None:
    """Updates the accessor object by parsing kwargs.

    Arguments:
        accessor_object: The accessor object (self).
        accessor_kwargs_dict: A TypedDict storing usable kwargs and types.
        kwargs_dict: The kwargs passed via get_xarray_data().

    Returns:
        None - this should be used to updated the accessor object.
    """
    # if kwargs are buried, dig them out
    while 'kwargs' in kwargs_dict.keys():
        kwargs_dict = kwargs_dict['kwargs']

    # get the TypedDict as a normal dict
    accessor_kwargs_dict = accessor_kwargs_dict.__annotations__

    # apply all kwargs that are in the TypedDict and match the type
    for key, value in kwargs_dict.items():
        if key not in accessor_kwargs_dict.keys():
            warnings.warn(
                f'Kwarg: {key} is allowed valid for {accessor_object.__name__}.'
            )
        elif not isinstance(value, accessor_kwargs_dict[key]):
            warnings.warn(
                f'Kwarg: {key} should be of type {accessor_kwargs_dict[key]}.'
            )
        else:
            setattr(accessor_object, key, value)


def combine_variables(
    dataset_dict: Dict[str, xr.Dataset],
    attrs_dict: Dict[str, Union[str, Number]],
    epsg: int = 4326,
) -> xr.Dataset:
    """Combines all variables into a single dataset."""
    # remove and warn about NoneType responses
    del_keys = []
    for k, v in dataset_dict.items():
        if v is None:
            del_keys.append(k)
        else:
            # write new metadata
            dataset_dict[k].attrs = attrs_dict

    if len(del_keys) > 0:
        warnings.warn(
            f'Could not get data for the following variables: {del_keys}'
        )
        for k in del_keys:
            dataset_dict.pop(k)

    # if just one variable, return the dataset
    if len(dataset_dict) == 0:
        raise ValueError(
            f'A problem occurred! No data was returned.'
        )

    # combine the data from multiple sources
    logging.info('Combining all variable Datasets...')
    out_ds = xr.merge(
        list(dataset_dict.values()),
    ).rio.write_crs(epsg)
    logging.info('Done! Returning combined dataset.')
    return out_ds
