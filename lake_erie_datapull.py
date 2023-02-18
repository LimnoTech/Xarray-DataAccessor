import read_into_xarray
from read_into_xarray import DataAccessor
from pathlib import Path
import pandas as pd
import gc
import logging

LAKE_ERIE_DIR = Path(Path.cwd() / 'lake_erie_data/')
AOI_SHP = Path(LAKE_ERIE_DIR / 'LEEM_boundary.shp')
RESOLUTION = 100  # meters
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
    'jan': ('1/01', '1/31'),
    'feb': ('2/01', '2/28'),
    'mar': ('3/01', '3/31'),
    'apr': ('4/01', '4/30'),
    'may': ('5/01', '5/31'),
    'jun': ('6/01', '6/30'),
    'jul': ('7/01', '7/31'),
    'aug': ('8/01', '8/31'),
    'sep': ('9/01', '9/30'),
    'oct': ('10/01', '10/31'),
    'nov': ('11/01', '11/30'),
    'dec': ('12/01', '12/31'),

}

# get data for a year at a time to avoid killing our memory


def main():
    for i, var_list in enumerate([ELEMENT_VARS, NODES_VARS]):
        logging.info(f'Getting data for {var_list}')
        for year in range(2011, 2022):
            for name, months_chunk in MONTH_CHUNKS.items():
                logging.info(
                    f'Getting data for year={year}, month range={name}')
                start_time = f'{months_chunk[0]}/{year}'
                end_time = f'{months_chunk[1]}/{year}'

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
                    resolution_factor=10,
                    chunk_dict={'time': 5},
                )

                # convert data_to_table
                logging.info('Saving data to .parquet')
                prefix = f'{name}_{year}_'
                data_accessor.get_data_tables(
                    variables=var_list,
                    csv_of_coords=pd.read_csv(CSVS[i]).loc[:100],
                    coords_id_column=COL_IDS[i],
                    save_table_dir=OUT_DIRS[i],
                    save_table_prefix=prefix,
                )

                # remove temp files
                data_accessor.unlock_and_clean()

                # delete the data accessor and clear memory
                del data_accessor
                gc.collect()


if __name__ == '__main__':
    main()
