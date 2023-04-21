"""Here we provide info/variables look up for the NASA data accessors.

Supported NASA providers (Data Access Centers) --------------------------------
Provider: LP DAAC = Land Processes Distributed Active Archive Center.
    Info: LP DAAC is a partnership between USGS and NASA. It is a component of the 
        Earth Observing System Data and Information System (EOSDIS).
    Data Source: Data is pulled from the LP DAAC Data Pool.
        See: https://lpdaac.usgs.gov/tools/data-pool/
    Data Access: One must have an Earthdata Login account to access the data.
        Sign up here: https://urs.earthdata.nasa.gov/users/new

Provider: PO DAAC = Physical Oceanography Distributed Active Archive Center.
    Info: PO DAAC is a NASA mission. It is a component of the
        Earth Observing System Data and Information System (EOSDIS).
    Data Source: Data is pulled from the PO DAAC Cloud hosted OPeNDAP server.
        See: https://opendap.jpl.nasa.gov/opendap/
    Data Access: One must have an Earthdata Login account to access the data.
        Sign up here: https://urs.earthdata.nasa.gov/users/new
"""
# datsets to check out below:
# https://lpdaac.usgs.gov/products/eco3etptjplv001/
# https://lpdaac.usgs.gov/products/glchmtv001/

# keep track of the dataset variables
LPDAAC_VARIABLES = {
    'NASADEM_NC': ['DEM'],
    'NASADEM_SC': [
        'slope',
        'aspect',
        'plan',
        'profile',
        'swbd',
    ],
    'GLanCE30': [
        'LC',
        # ignoring others. Vastly complicates CRM search behavior.
        # 'ChgDate',
        # 'PrevClass',
        # 'EVI2med',
        # 'EVIamp',
        # 'EVI2rate',
        # 'EVI2chg',
    ],
}

# keeps track of which datasets have time dimensions
# NOTE: if not None, these much match datetime object attribute conventions
LPDAAC_TIME_DIMS = {
    'NASADEM_NC': None,
    'NASADEM_SC': None,
    'GLanCE30': 'year',
}

LPDAAC_XY_DIMS = {
    'NASADEM_NC': ['lon', 'lat'],
    # 'NASADEM_SC': ['lat', 'lon'],
    'GLanCE30': ['x', 'y'],
}

# NOTE: GLanCE30 is in a projected CRS - https://measures-glance.github.io/glance-grids/params.html
LPDAAC_EPSG = {
    'NASADEM_NC': 4326,
    'NASADEM_SC': 4326,
    'GLanCE30': None
}
LPDAAC_WKT = {
    'NASADEM_NC': None,
    'NASADEM_SC': None,
    'GLanCE30': 'PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - NA - V01",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["longitude_of_center",-100],PARAMETER["latitude_of_center",50],UNIT["meter",1.0]]'
}
