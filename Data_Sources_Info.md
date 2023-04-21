Data Sources Info
==================

## *ERA5 Global Climate Model Data Accessors*
### **Copernicus Data Store API - `CDSDataAccessor`**
[API Documentation](https://cds.climate.copernicus.eu/)

**Note:** To use the CDS API, one must save a "dot-file" with their API key on it after making and logging into their account. Specific instructions are available [here](https://cds.climate.copernicus.eu/api-how-to). One can ignore the instructions about installing `cdsapi` as that is included in this library's `environment.yml` file.

Single levels (hourly) datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
* reanalysis-era5-single-levels
* reanalysis-era5-single-levels-preliminary-back-extension

Single levels (monthly) datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview) (not supported as of 4/5/2023)
* reanalysis-era5-single-levels-monthly-means
* reanalysis-era5-single-levels-monthly-means-preliminary-back-extension

Pressure levels datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) (not supported as of 4/5/2023)
* reanalysis-era5-pressure-levels
* reanalysis-era5-pressure-levels-monthly-means
* reanalysis-era5-pressure-levels-preliminary-back-extension
* reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension

ERA5-land datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)
* reanalysis-era5-land
* reanalysis-era5-land-monthly-means

### **Planet OS AWS bucket - `AWSDataAccessor`**
ERA-5 single/surface level data is also available on an [AWS s3 bucket](https://aws.amazon.com/marketplace/pp/prodview-yhz3mavy6s7go#similar-products). However, there are far fewer variables available, and the data goes back to 1979 (as opposed to 1959).

Single levels (hourly) datasets - [info](https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md)
* reanalysis-era5-single-levels

Note that the variables are named differently. To see the variable names available one can run the following function and be returned a list:
```python
from xarray_data_accessor import DataAccessorFactory

DataAccessorFactory.supported_variables(
    data_accessor_name='AWSDataAccessor',
    dataset_name='reanalysis-era5-single-levels',
)
```

A crosswalk between CDS and AWS variables names can be found [here](https://github.com/LimnoTech/Xarray-DataAccessor/blob/main/src/xarray_data_accessor/data_accessors/era5_from_cds_info.py#L40).

### A note on `CDSDataAccessor` vs `AWSDataAccessor`

While reading data from AWS can be much faster than the CDS API (especially for large time ranges), loading the AWS data to disk is much slower! 

We recommend using `AWSDataAccessor` for data visualization and xarray native workflows. However, if you want to sample the data and convert into a pandas data frame (i.e., via `xarray_data_accessor.get_data_tables()`), using `CDSDataAccessor` will be significantly faster.

Another relevant difference is that AWS ERA5 data is returned along a uniform 0.25 decimal degree grid (i.e., 0.25, 0.5, 0.75,...) while CDS returns a grid with 0.25 increments as well, but centered based on the bounding box.

## NASA DataAccessors
**Note:** For all NASA DataAccessors one must have an active [EarthData Account]( https://urs.earthdata.nasa.gov/users/new), and pass in your username/password via the following `get_xarray_dataset()` keyword argument `authorization={'username': 'example_username', 'password': 'example_password'}`.


### NASA/USGS LP DAAC DataPool - `NASA_LPDAAC_Accessor`
[Organization information](https://lpdaac.usgs.gov/about/).

NASADEM_NC - [info](https://lpdaac.usgs.gov/products/nasadem_hgtv001/)
* This dataset provides a global 30m Digital Elevation Model derived from the Shuttle Radar Topography Mission (SRTM). Access to the QA/QC layers documented in the attached link ("NUM" and "SWB") are not currently supported, the only associated variable is "DEM" providing elevation relative to sea level in meters.

NASADEM_SC - [info](https://lpdaac.usgs.gov/products/nasadem_scv001/)
* This dataset provides DEM by-products. See the "Layers" section of the dataset documentation for details.

**Note:** Both of the NASADEM datasets have no time dimension. However, one must still provide a start_time/end_time argument to `get_xarray_dataset()`. The time provided will not effect the data pulled.

GLanCE30 - [info](https://lpdaac.usgs.gov/products/glance30v001/)
* This dataset provides a (mostly) global 7 class Land Cover (LC) grid at 30m / yearly resolution. Note that there are areas without coverage. The "LC" variable provides the main grid, however other documented variables can be accessed that track land cover class changes over time.
* **Note:** All GLanCE30 data is collected on July 1st (7/1), therefore if your date range does not pass over July no data will be returned!

