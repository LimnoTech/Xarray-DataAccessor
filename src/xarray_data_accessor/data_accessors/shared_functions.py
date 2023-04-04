import logging
import warnings
import xarray as xr
from typing import (
    Dict,
    Union,
    Number,
)


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
