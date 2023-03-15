import read_into_xarray
from read_into_xarray import DataAccessor
from pathlib import Path
import pandas as pd
import gc
import logging
logging.basicConfig(
    filename='Lake_Erie_data_pull.log',
    level=logging.INFO,
)

LAKE_ERIE_DIR = Path(Path.cwd() / 'lake_erie_data/')
AOI_SHP = Path(LAKE_ERIE_DIR / 'LEEM_boundary.shp')
RESOLUTION = 50  # meters
DATASET_NAME = 'reanalysis-era5-single-levels'
VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    'total_cloud_cover',
    'mean_surface_downward_short_wave_radiation_flux',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
]

# set up directories for elements and nodes
NODES_DIR = Path(LAKE_ERIE_DIR / 'nodes_data')
if not NODES_DIR.exists():
    NODES_DIR.mkdir()
ELEMENTS_DIR = Path(LAKE_ERIE_DIR / 'elements_data')
if not ELEMENTS_DIR.exists():
    ELEMENTS_DIR.mkdir()

NODES_VARS = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    'total_cloud_cover',
    'mean_surface_downward_short_wave_radiation_flux',
]
ELEMENT_VARS = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
]

COL_IDS = ['NELE', 'NODE']
CSVS = [
    Path(LAKE_ERIE_DIR / 'element_latlongs.csv'),
    Path(LAKE_ERIE_DIR / 'node_latlongs.csv'),
]
OUT_DIRS = [ELEMENTS_DIR, NODES_DIR]
MONTH_CHUNKS = {
    'jan': ('1/01', '2/01'),
    'feb': ('2/01', '3/01'),
    'mar': ('3/01', '4/01'),
    'apr': ('4/01', '5/01'),
    'may': ('5/01', '6/01'),
    'jun': ('6/01', '7/01'),
    'jul': ('7/01', '8/01'),
    'aug': ('8/01', '9/01'),
    'sep': ('9/01', '10/01'),
    'oct': ('10/01', '11/01'),
    'nov': ('11/01', '12/01'),
    'dec': ('12/01', '1/01'),
}


def reorder_columns(df: pd.DataFrame) -> pd.Index:
    """
    Returns columns reordered where datetime is the first, and the rest are numeric.

    NOTE: this is specific for this exact use case.

    """
    # reorder the integer columns and turn back to strings
    int_cols = df.columns[1:].astype('int').sort_values()
    reordered_ints = int_cols.astype('str')
    del int_cols
    del df

    # use Index.append() to add datetime back to the start
    return pd.Index(['datetime']).append(reordered_ints)


def convert_to_csvs(
    data_dir: Path,
    var_names: list[str],
    months: list[str],
    csv_prefix: str,
    reorder: bool = True,
) -> None:
    """Combines parquets into a single massive CSV"""
    logging.info(f'Combining all .parquet files in {data_dir}')
    var_parquets = {}
    for var in var_names:
        ordered_parquets = []
        for year in range(2011, 2022):
            for month in months:
                ordered_parquets.append(
                    data_dir / f'{month}_{year}_{var}.parquet'
                )
        var_parquets[var] = ordered_parquets

    for var in list(var_parquets.keys()):
        parquets = var_parquets[var]
        out_path = data_dir / f'{csv_prefix}_{var}.csv',

        logging.info(f'Making {out_path}')
        var_df = None
        columns_order = None

        # combine all the parquet files into one DataFrame
        for i, parquet in enumerate(parquets):

            # get updated column order if desired
            if reorder and columns_order is None:
                columns_order = reorder_columns(
                    pd.read_parquet(
                        parquet,
                        engine='pyarrow',
                    )
                )

            # read in data
            df = pd.read_parquet(
                parquet,
                engine='pyarrow',
                columns=columns_order,
            )
            if i == 0:
                var_df = df.copy()
            else:
                var_df = pd.concat([var_df, df])
            del df
            print(f'{i}/{len(parquets)} - Shape: {var_df.shape}')
        print('Saving to CSV')
        var_df.to_csv(
            out_path,
            chunksize=2000,
        )
        del var_df


def main():
    for i, var_list in enumerate([ELEMENT_VARS, NODES_VARS]):
        logging.info(f'Getting data for {var_list}')

        # iterate over years and months
        for year in range(2011, 2022):
            for name, months_chunk in MONTH_CHUNKS.items():

                # adjust for december to make sure we are not missing days
                if name != 'dec':
                    end_year = year
                else:
                    end_year = year + 1

                # get start and end times (note that the end is exclusive!)
                start_time = f'{months_chunk[0]}/{year}'
                end_time = f'{months_chunk[1]}/{end_year}'

                # make sure we aren't overwriting
                prefix = f'{name}_{year}_'
                done_files = Path(OUT_DIRS[i]).iterdir()
                done = False
                for file in done_files:
                    if prefix in file.name:
                        done = True
                if not done:
                    logging.info(
                        f'Getting data for year={year}, month range={name}'
                    )

                # init our data accessor
                    data_accessor = DataAccessor(
                        dataset_name=DATASET_NAME,
                        variables=var_list,
                        start_time=start_time,
                        end_time=end_time,
                        coordinates=None,
                        csv_of_coords=None,
                        shapefile=AOI_SHP,
                        raster=None,
                        multithread=True,
                    )

                    # get the data
                    logging.info(f'Getting and resampling data')
                    data_accessor.get_data(
                        resolution_factor=RESOLUTION,
                        use_cds_only=True,
                    )

                    # convert data_to_table
                    logging.info('Saving data to .parquet')
                    data_accessor.get_data_tables(
                        variables=var_list,
                        csv_of_coords=pd.read_csv(CSVS[i]),
                        coords_id_column=COL_IDS[i],
                        save_table_dir=OUT_DIRS[i],
                        save_table_prefix=prefix,
                    )

                    # remove temp files
                    data_accessor.unlock_and_clean()

                    # delete the data accessor and clear memory
                    del data_accessor
                    gc.collect()

    # combine ELEMENTS data into one giant CSV file
    convert_to_csvs(
        ELEMENTS_DIR,
        ELEMENT_VARS,
        list(MONTH_CHUNKS.keys()),
        'ELEMENTS',
        reorder=True,
    )

    # combine NODES data into one giant CSV file
    convert_to_csvs(
        NODES_DIR,
        NODES_VARS,
        list(MONTH_CHUNKS.keys()),
        'NODES',
        reorder=True,
    )


logging.shutdown()
if __name__ == '__main__':
    main()
