import itertools
import xarray as xr
import pandas as pd
import numpy as np
import xarray_data_accessor.utility_functions as utility_functions
from xarray_data_accessor.data_converters.base import DataConverterBase
from xarray_data_accessor.shared_types import (
    CoordsTuple,
    TableInput,
)
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


class ConvertToTable(DataConverterBase):
    """Contains functions to convert xarray datasets to tables."""

    @staticmethod
    def points_to_tables(
        xarray_dataset: xr.Dataset,
        variables: Optional[List[str]] = None,
        coords: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[TableInput] = None,
        coords_id_column: Optional[str] = None,
        xy_columns: Optional[Tuple[str, str]] = None,
        save_table_dir: Optional[Union[str, Path]] = None,
        save_table_suffix: Optional[str] = None,
        save_table_prefix: Optional[str] = None,
    ) -> Dict[str, Union[pd.DataFrame, Path]]:
        """
        Returns:
            A dictionary with variable names as keys, and dataframes as values
                if save_table_dir==None, or the output table path as values if
                save_table_dir is not None.
        """
        # init output dictionary
        out_dict = {}

        # clean variables input
        variables = utility_functions._verify_variables(
            xarray_dataset,
            variables,
        )

        # get x/y columns
        if xy_columns is None:
            xy_columns = ('lon', 'lat')
        x_col, y_col = xy_columns

        # get coords input from csv
        coords_df = utility_functions._get_coords_df(
            coords=coords,
            csv_of_coords=csv_of_coords,
            coords_id_column=coords_id_column,
        )

        # get the point x/y values
        point_xs = coords_df[x_col].values
        point_ys = coords_df[y_col].values
        point_ids = [str(i) for i in coords_df.index.values]

        # get dimension names
        x_dim = xarray_dataset.attrs['x_dim']
        y_dim = xarray_dataset.attrs['y_dim']

        # get all coordinates from the dataset
        ds_xs = xarray_dataset[x_dim].values
        ds_ys = xarray_dataset[y_dim].values

        # get nearest lat/longs for each sample point
        nearest_x_idxs = np.abs(ds_xs - point_xs.reshape(-1, 1)).argmin(axis=1)
        nearest_y_idxs = np.abs(ds_ys - point_ys.reshape(-1, 1)).argmin(axis=1)

        # get a dict with point IDs as keys, and nearest x/y indices as values
        points_nearest_xy_idxs = dict(
            zip(
                point_ids,
                zip(nearest_x_idxs, nearest_y_idxs),
            ),
        )

        # get all x/long to y/lat combos
        combos = list(
            itertools.product(
                range(len(xarray_dataset[x_dim].values)),
                range(len(xarray_dataset[y_dim].values)),
            ),
        )

        # make sure they are in the right order to reshape!
        combo_dict = dict(zip(combos, range(len(combos))))

        # get point id to xy combo index
        id_to_index = {}
        for pid, coord in points_nearest_xy_idxs.items():
            id_to_index[pid] = combo_dict[coord]

        # clear some memory
        del (
            nearest_x_idxs,
            nearest_y_idxs,
            point_xs,
            point_ys,
            combos,
            combo_dict,
        )

        # prep chunks
        xarray_dataset = xarray_dataset.chunk(
            {'time': 10000, x_dim: 10, y_dim: 10},
        )

        # get data for each variable
        for variable in variables:
            out_dict[variable] = utility_functions._get_data_table_vectorized(
                xarray_dataset,
                variable,
                point_ids,
                id_to_index,
                xy_dims=(x_dim, y_dim),
                save_table_dir=save_table_dir,
                save_table_suffix=save_table_suffix,
                save_table_prefix=save_table_prefix,
            )
        return out_dict

    @classmethod
    def get_conversion_functions(
        cls,
    ) -> Dict[str, DataConverterBase.ConversionFunctionType]:
        """Returns a dictionary of conversion functions."""
        return {
            cls.points_to_tables.__name__: cls.points_to_tables,
        }
