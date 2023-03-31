import xarray_data_accessor
from xarray_data_accessor import DataAccessor
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

START_TIME = '12/15/2020'
END_TIME = '12/31/2020'
AOI_SHP = Path(Path.cwd() / 'lake_erie_data/LEEM_boundary.shp')
DATASET_NAME = 'reanalysis-era5-single-levels'
VARIABLES = [
    '2m_temperature',
    # '2m_dewpoint_temperature',
    'total_cloud_cover',
    # 'mean_surface_downward_short_wave_radiation_flux',
    # '10m_u_component_of_wind',
    # '10m_v_component_of_wind',
]
LAKE_ERIE_DIR = Path(Path.cwd() / 'lake_erie_data/')
NODES_CSV = Path(LAKE_ERIE_DIR / 'node_latlongs.csv')
POINT_AMOUNT = 1000


def main():
    # get coords as a dataframe and sample 100 rows randomly
    coords_df = pd.read_csv(NODES_CSV)
    coords_df = coords_df.loc[np.random.randint(
        coords_df.shape[0], size=POINT_AMOUNT)]
    coords_df.info()
    coords_df.head()

    data_accessor = DataAccessor(
        dataset_name=DATASET_NAME,
        variables=VARIABLES,
        start_time=START_TIME,
        end_time=END_TIME,
        coordinates=None,
        csv_of_coords=None,
        shapefile=AOI_SHP,
        raster=None,
        multithread=True,
    )

    ds = data_accessor.get_data(
        use_cds_only=True,
        resolution_factor=25,
    )
    tables = data_accessor.get_data_tables(
        csv_of_coords=coords_df,
        coords_id_column='NODE',
        save_table_dir=Path.cwd(),
        save_table_suffix='.csv',
    )
    print(ds.info)


if __name__ == '__main__':
    main()
