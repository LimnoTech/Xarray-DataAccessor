from read_into_xarray import DataAccessor
import logging
import gc
from pathlib import Path

COORDS = [(-5.141658, -40.916180)]
START_TIME = '01/01/2017'
END_TIME = '12/31/2022'
VARIABLES = [
    'convective_precipitation',
]
BRAZIL_DIR = Path.cwd() / 'brazil_data'

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


def main():
    for year in range(2017, 2023):
        for name, months_chunk in MONTH_CHUNKS.items():
            logging.info(
                f'Getting data for year={year}, month range={name}')
            start_time = f'{months_chunk[0]}/{year}'
            end_time = f'{months_chunk[1]}/{year}'

            # init our data accessor
            data_accessor = DataAccessor(
                dataset_name='reanalysis-era5-single-levels',
                variables=VARIABLES,
                start_time=start_time,
                end_time=end_time,
                coordinates=COORDS,
                csv_of_coords=None,
                shapefile=None,
                raster=None,
                multithread=True,
            )

            # get the data
            logging.info(f'Getting and resampling data')
            data_accessor.get_data()

            # convert data_to_table
            logging.info('Saving data to .parquet')
            prefix = f'{name}_{year}_'
            data_accessor.get_data_tables(
                variables=VARIABLES,
                csv_of_coords=Path(BRAZIL_DIR / 'brazil_coords.csv'),
                coords_id_column='station',
                save_table_dir=BRAZIL_DIR,
                save_table_prefix=prefix,
            )

            # remove temp files
            data_accessor.unlock_and_clean()

            # delete the data accessor and clear memory
            del data_accessor
            gc.collect()


if __name__ == '__main__':
    main()
