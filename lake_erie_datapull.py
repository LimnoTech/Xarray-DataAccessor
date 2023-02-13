from read_into_xarray import DataAccessor
from pathlib import Path
"""
Hi Xavier,

Per our discussion a bit ago, we would like you to extract the climate model output at each of our model cell locations.  As I mentioned, some of the variables we need are assigned to “elements” (triangular cells) and some are assigned to “nodes” (vertices of the triangular cells).  Both of those model shapefiles are located in the paths below:

Wind components are assigned at “elements”:       
I:\LEEM\GIS\data\FVCOM\Grids\Fine\FVCOM_Elements.shp
Air temp, cloud cover, dew point temp and shortwave radiation are assigned at “nodes”:  
I:\LEEM\GIS\data\FVCOM\Grids\Fine\FVCOM_Nodes.shp
Lake Erie boundary shapefile:  I:\LEEM\GIS\data\FVCOM\Grids\Fine\LEEM_boundary.shp

Also, the attached excel file contains the lat and lon for all of the locations for both shapefiles.  

For the spatial resampling/interpolation, it looks like the smallest triangular cells are on the order of 0.1 km2, so we don’t need to get super fine there.  Use your best judgement on what is feasible for computer resources.

Hourly output would be ideal.  And ultimately, if we find this climate data to be useful, we would need 2011-2021, but starting with a shorter time period is fine.  Lets say 9/1/2014-10/1/2016.  If that’s too much data, then try 9/1/2015-10/1/2016?

Thanks a bunch

-Dan
"""
START_TIME = '9/1/2015'
END_TIME = '10/1/2016'
AOI = Path('lake_erie_data/LEEM_boundary.shp')
RESOLUTION = 100 # meters
air temp
dew point
cloud cover
short wave radiation
u (eastward) component of wind speed
v (northward) component of wind speed

VARIABLES = [

]

data_accessor = DataAccessor(
    dataset_name='reanalysis-era5-single-levels',
    variables: Union[str, List[str]],
    start_time=START_TIME,
    end_time=END_TIME,
    coordinates=None,
    csv_of_coords=None,
    shapefile: Optional[Union[str, Path]] = None,
    raster=None,
    multithread: bool = True,
)
