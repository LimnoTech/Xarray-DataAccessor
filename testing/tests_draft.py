"""A place to put tests that are not ready to be run yet."""
import xarray_data_accessor
from pathlib import Path

TEST_DATA = Path.cwd() / 'testing/test_data'


def main(
    test_cds: bool = True,
    test_aws: bool = True,
):
    # get a few days of data crossing a month boundary
    if test_cds:
        cds_era5_dataset = xarray_data_accessor.get_xarray_dataset(
            data_accessor_name='CDSDataAccessor',
            dataset_name='reanalysis-era5-single-levels',
            variables=['2m_temperature', '100m_u_component_of_wind'],
            start_time='2019-01-30',
            end_time='2019-02-02',
            shapefile=Path(TEST_DATA / 'LEEM_boundary.shp'),
        )

    if test_aws:
        aws_era5_xarray = xarray_data_accessor.get_xarray_dataset(
            data_accessor_name='AWSDataAccessor',
            dataset_name='reanalysis-era5-single-levels',
            variables=['air_temperature_at_2_metres',
                       'eastward_wind_at_100_metres'],
            start_time='2019-01-30',
            end_time='2019-02-02',
            shapefile=Path(TEST_DATA / 'LEEM_boundary.shp'),
        )
    print(xarray_data_accessor.__version__)


if __name__ == '__main__':
    main(
        test_cds=True,
        test_aws=True,
    )
