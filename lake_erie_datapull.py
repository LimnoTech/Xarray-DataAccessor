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
ELEMENTS_DIR = Path(LAKE_ERIE_DIR / 'nodes_data')
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

# get data for a year at a time to avoid killing our memory
for year in range(2011, 2022):
    start_time = f'9/1/{year}'
    end_time = f'9/30/{year + 1}'

    # init our data accessor
    data_accessor = DataAccessor(
        dataset_name=DATASET_NAME,
        variables=VARIABLES,
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
        resolution_factor=30,
        chunk_dict={'time': 5},
    )

    # convert data_to_table for NODES
    out_dir = Path(NODES_DIR / f'water_year_{year}to{year + 1}_data')
    if not out_dir.exists():
        out_dir.mkdir()
    data_accessor.get_data_tables(
        variables=NODES_VARS,
        save_table_dir=out_dir
    )

    # convert data_to_table for ELEMENTS
    out_dir = Path(ELEMENTS_DIR / f'water_year_{year}to{year + 1}_data')
    if not out_dir.exists():
        out_dir.mkdir()
    data_accessor.get_data_tables(
        variables=ELEMENT_VARS,
        save_table_dir=out_dir
    )

    # remove temp files
    data_accessor.unlock_and_clean()

    # delete the data accessor and clear memory
    del data_accessor
    gc.collect()
