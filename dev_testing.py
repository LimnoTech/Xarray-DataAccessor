import read_into_xarray
from read_into_xarray import DataAccessor
from pathlib import Path
import geopandas as gpd

START_TIME = '12/15/2020'
END_TIME = '12/31/2020'  # NOT DANIELS TIME AOI
AOI_SHP = Path(Path.cwd() / 'lake_erie_data/LEEM_boundary.shp')
DATASET_NAME = 'reanalysis-era5-single-levels'
VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    'total_cloud_cover',
    'mean_surface_downward_short_wave_radiation_flux',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
]


def main():
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

    ds = data_accessor.get_data()


if __name__ == '__main__':
    main()
