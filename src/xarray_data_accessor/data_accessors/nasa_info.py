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
        'ChgDate',
        'PrevClass',
        'EVI2med',
        'EVIamp',
        'EVI2rate',
        'EVI2chg',
    ],
}

# keeps track of which datasets have time dimensions
LPDAAC_TIME_DIMS = {
    'NASADEM_NC': False,
    'NASADEM_SC': False,
    'GLanCE30': True,
}
