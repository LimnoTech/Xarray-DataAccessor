# ERA5-to-Xarray

**NOTE:** Change name to something more generic!


A repo to efficiently read data into [Xarray](https://docs.xarray.dev/en/stable/) using [dask.distributed](https://distributed.dask.org/en/stable/) for parallelization via the CDS API or s3 AWS bucket.

Based on [`era5cli`](https://github.com/eWaterCycle/era5cli)'s source code.

**Features to build:**
* CDS API call object.
* An object that inherits from that for each relevant dataset.
* support pressure levels query

# Datasets 
## Copernicus Data Store API
[**API Documentation**](https://cds.climate.copernicus.eu/)

**Single levels (hourly) datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)**
* reanalysis-era5-single-levels
* reanalysis-era5-single-levels-preliminary-back-extension

**Single levels (monthly) datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview)**
* reanalysis-era5-single-levels-monthly-means
* reanalysis-era5-single-levels-monthly-means-preliminary-back-extension

**Pressure levels datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)**
* reanalysis-era5-pressure-levels
* reanalysis-era5-pressure-levels-monthly-means
* reanalysis-era5-pressure-levels-preliminary-back-extension
* reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension

**ERA5-land datasets - [info](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)**
* reanalysis-era5-land
* reanalysis-era5-land-monthly-means

## Planet OS AWS bucket
ERA-5 single/surface level data is available on an [AWS s3 bucket](https://aws.amazon.com/marketplace/pp/prodview-yhz3mavy6s7go#similar-products). However, there are far fewer variables available, and the data goes back to 1979 (as opposed to 1959).

By default, data will be pulled from the AWS bucket if possible since it allows one to avoid saving large NetCDF files.

See the Planet OS [documentation](https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md) for a list of available variables.
