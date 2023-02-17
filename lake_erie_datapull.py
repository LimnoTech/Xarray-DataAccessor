import read_into_xarray
from read_into_xarray import DataAccessor
from pathlib import Path
import gc

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
MONTH_CHUNKS = ['jan_thru_june', 'july_thru_dec']

# get data for a year at a time to avoid killing our memory
for i, var_list in enumerate([ELEMENT_VARS, NODES_VARS]):
    for year in range(2011, 2022):
        for j, months_chunk in enumerate([('1/01', '6/30'), ('7/01', '12/31')]):
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
            data_accessor.get_data(
                resolution_factor=20,
                chunk_dict={'time': 5},
            )

            # convert data_to_table for NODES
            prefix = f'{MONTH_CHUNKS[j]}_{year}_'
            data_accessor.get_data_tables(
                variables=var_list,
                csv_of_coords=CSVS[i],
                coords_id_column=COL_IDS[i],
                save_table_dir=OUT_DIRS[i],
                save_table_prefix=prefix,
            )

            # remove temp files
            data_accessor.unlock_and_clean()

            # delete the data accessor and clear memory
            del data_accessor
            gc.collect()
