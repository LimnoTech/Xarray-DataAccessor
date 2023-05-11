Xarray-DataAccessor Documentation
==================================

## Core Features
* Efficiently reads remote gridded data for an Area of Interest (AOI) into [Xarray.Dataset](https://docs.xarray.dev/en/stable/) objects using [dask.distributed](https://distributed.dask.org/en/stable/) for parallelization.
* Transform data for your needs: resample the grid, resample along a time dimension, convert timezone, etc.
* Extract time series data at coordinates and save to a tabular file (i.e. .xlsx, .csv, or .parquet) for use in physical or machine learning models.
* Extendable/modular package architecture supporting open-source contributions, and connections to more datasets/sources.

## Getting Started
1. Start by cloning this repository locally.
2. Next, within an conda terminal navigate to the local repository location and clone and activate our conda virtual environment using `environment.yml`.
```
# mock conda terminal
(base) C://User: cd Path/To/Xarray-DataAccessor
(base) C://User/Path/To/Xarray-DataAccessor conda env create -f environment.yml
...
(base) C://User/Path/To/Xarray-DataAccessor conda activate xarray_data_accessor_env
(xarray_data_accessor_env) C://User/Path/To/Xarray-DataAccessor
```
3. (optional) if you plan to use the `CDSDataAccessor`, follow the instructions [here](https://cds.climate.copernicus.eu/api-how-to) to allow your computer to interact with the CDS API. Basically you must manually make a `.cdsapirc` text file (no extension!) where `cdsapi.Client()` expects it to be.
4. Use the [conda-develop](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) `build` command pointed to the `/src/` directory to make the repo importable.
```
# mock conda terminal with the env activated
(xarray_data_accessor_env) C://User/Path/To/Xarray-DataAccessor conda-build src

# a this point you are ready to open an IDE/Notebook of your choice to run your code!
# For example:
(xarray_data_accessor_env) C://User/Path/To/Xarray-DataAccessor jupyterlab
```
5. Finally, import the library into your workflow:
```python
import xarray_data_accessor
```

## Exploring Available Data
All data one can retrieve with this library is organized in a three tier hierarchy:
1. A "data accessor" is a python class that interacts with a given data source.
    * Each data accessor can retrieve data from any number of specific datasets.
    * For example: `CDSDataAccessor` accesses the [CDS API](https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset) and can currently be used to access a few ERA5 datasets.
2. A specific dataset may be something like "reanalysis-era5-single-levels". Note that the same dataset may be able to be accessed by different data accessors.
3. Each dataset will contain one or more variables.

To allow this library to be extendable, the "data accessors", the datasets they can access, and the variables that exist in each dataset are not hardcoded anywhere in the repo.

**Therefore to explore what is available, one can use the following `xarray_data_accessor.DataAccessorFactory` class functions:**
```python
from xarray_data_accessor import DataAccessorFactory

# to return a list of all data accessor names
DataAccessorFactory.data_accessor_names()

# to return a dictionary with data accessor names as keys and their respective objects and values
DataAccessorFactory.data_accessor_objects()

# to return a dictionary with data accessor names as keys, and their supported dataset names as values
DataAccessorFactory.supported_datasets()

# to return a list of variable names for a specific data accessor - dataset combination
DataAccessorFactory.supported_variables(
    data_accessor_name: str,
    dataset_name: str,
)
```

We also intend to keep documentation about data accessors and their respective datasets updated [here](https://github.com/LimnoTech/Xarray-DataAccessor/blob/main/Data_Sources_Info.md).

## Getting Data
To get data one can use the `get_xarray_dataset()` function after specifying time and space AOI.

The spatial AOI can be specified with a shapefile, raster, a list of lat/long coordinate tuples, or a csv with lat/lon as columns.

The temporal AOI can be specified as a string or a datetime object. Additionally, one can specify a timezone using `param:timezone`.

In the example below we fetch ERA5 data from AWS for a shapefile defined extent.
```python
import xarray_data_accessor
dataset = xarray_data_accessor.get_xarray_dataset(
        data_accessor_name='AWSDataAccessor',
        dataset_name='reanalysis-era5-single-levels',
        variables=[
            'air_temperature_at_2_metres',
            'eastward_wind_at_100_metres',
        ],
        start_time='2019-01-30',
        end_time='2019-02-02',
        shapefile='path/to/shapefile.shp',
    )
```


## Transforming Data
Functionality has not been thoroughly tested...documentation pending.

## Development Road Map

- [x] Build out base architecture and library design.
- [x] Build `CDSDataAccessor` to retrieve ERA5 hourly data from the CDS API.
- [x] Build `AWSDataAccessor` to retrieve ERA5 hourly data from the Planet OS S3 bucket.
- [x] Build a function to spatially resample data.
- [x] Build a function to convert data timezones.
- [x] Build a function to sample data across the time dimension and export to a table file.
- [x] Build a `pytest` test suite for the two ERA5 data accessors as well as the `DataAccessorFactory` class functions.
- [x] Set up documentation structure.
- [x] Build a `DataAccessorBase` implementation to NASA LP-DAAC data (elevation and land cover).
- [ ] Build a function to temporally resample data.
- [ ] Build a `pytest` test suite for all the data transformation functions.
- [ ] Build "Data Stacks" that align data from different sources such that it is ready for modelling.
- [ ] Build a `DataAccessorBase` implementation to fetch soils data (type and moisture).
- [ ] Make the package pip installable.
