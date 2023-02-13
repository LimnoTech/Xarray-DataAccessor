"""
Lists of ERA-5 variables and pressure levels.

NOTE: This was collected manually, as no API call for variable availability exists.

Documentation resources via
https://cds.climate.copernicus.eu/

All links below prefaced with:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/

AWS single levels (hourly) data variables have different names and are stored
    under AWS_VARIABLES_DICT. The standard CDS names are the dictionary keys,
    and the s3 bucket names are the values.
    s3 bucket docs: https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
    dataset = 'reanalysis-era5-single-levels'.

DATASETS 
--------------------------------

Single levels (hourly) variables
* reanalysis-era5-single-levels
* reanalysis-era5-single-levels-preliminary-back-extension

Single levels (monthly) variables
* reanalysis-era5-single-levels-monthly-means
* reanalysis-era5-single-levels-monthly-means-preliminary-back-extension

Pressure levels variables
* reanalysis-era5-pressure-levels
* reanalysis-era5-pressure-levels-monthly-means
* reanalysis-era5-pressure-levels-preliminary-back-extension
* reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension

ERA5-land variables
* reanalysis-era5-land
* reanalysis-era5-land-monthly-means
"""
from typing import List

# data sources and dataset names
DATASET_SOURCES = ('CDS', 'AWS')

DATASET_NAMES = [
    'reanalysis-era5-single-levels',
    'reanalysis-era5-single-levels-preliminary-back-extension',
    'reanalysis-era5-single-levels-monthly-means',
    'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension',
    'reanalysis-era5-pressure-levels',
    'reanalysis-era5-pressure-levels-monthly-means',
    'reanalysis-era5-pressure-levels-preliminary-back-extension',
    'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
    'reanalysis-era5-land',
    'reanalysis-era5-land-monthly-means',
]

AWS_VARIABLES_DICT = {
    '10m_u_component_of_wind': 'eastward_wind_at_10_metres',
    '10m_v_component_of_wind': 'northward_wind_at_10_metres',
    '100m_u_component_of_wind': 'eastward_wind_at_100_metres',
    '100m_v_component_of_wind': 'northward_wind_at_100_metres',
    '2m_dewpoint_temperature': 'dew_point_temperature_at_2_metres',
    '2m_temperature': 'air_temperature_at_2_metres',
    'maximum_2m_temperature_since_previous_post_processing': 'air_temperature_at_2_metres_1hour_Maximum',
    'minimum_2m_temperature_since_previous_post_processing': 'air_temperature_at_2_metres_1hour_Minimum',
    'mean_sea_level_pressure': 'air_pressure_at_mean_sea_level',
    'mean_wave_period': 'sea_surface_wave_mean_period',
    'mean_wave_direction': 'sea_surface_wave_from_direction',
    'significant_height_of_total_swell': 'significant_height_of_wind_and_swell_waves',
    'snow_density': 'snow_density',
    'snow_depth': 'lwe_thickness_of_surface_snow_amount',
    'surface_pressure': 'surface_air_pressure',
    'surface_solar_radiation_downwards': 'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
    'maximum_total_precipitation_rate_since_previous_post_processing': 'precipitation_amount_1hour_Accumulation',
}

# all single level variables in both hourly and monthly data
SINGLE_LEVEL_VARIABLES = [
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    '10m_u_component_of_neutral_wind',
    '10m_u_component_of_wind',
    '10m_v_component_of_neutral_wind',
    '10m_v_component_of_wind',
    '10m_wind_gust_since_previous_post_processing',
    '10m_wind_speed',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'air_density_over_the_oceans',
    'altimeter_corrected_wave_height',
    'altimeter_range_relative_correction',
    'altimeter_wave_height',
    'angle_of_sub_gridscale_orography',
    'anisotropy_of_sub_gridscale_orography',
    'benjamin_feir_index',
    'boundary_layer_dissipation',
    'boundary_layer_height',
    'charnock',
    'clear_sky_direct_solar_radiation_at_surface',
    'cloud_base_height',
    'coefficient_of_drag_with_waves',
    'convective_available_potential_energy',
    'convective_inhibition',
    'convective_precipitation',
    'convective_rain_rate',
    'convective_snowfall',
    'convective_snowfall_rate_water_equivalent',
    'downward_uv_radiation_at_the_surface',
    'duct_base_height',
    'eastward_gravity_wave_surface_stress',
    'eastward_turbulent_surface_stress',
    'evaporation',
    'forecast_albedo',
    'forecast_logarithm_of_surface_roughness_for_heat',
    'forecast_surface_roughness',
    'free_convective_velocity_over_the_oceans',
    'friction_velocity',
    'geopotential',
    'gravity_wave_dissipation',
    'high_cloud_cover',
    'high_vegetation_cover',
    'ice_temperature_layer_1',
    'ice_temperature_layer_2',
    'ice_temperature_layer_3',
    'ice_temperature_layer_4',
    'instantaneous_10m_wind_gust',
    'instantaneous_eastward_turbulent_surface_stress',
    'instantaneous_large_scale_surface_precipitation_fraction',
    'instantaneous_moisture_flux',
    'instantaneous_northward_turbulent_surface_stress',
    'instantaneous_surface_sensible_heat_flux',
    'k_index',
    'lake_bottom_temperature',
    'lake_cover',
    'lake_depth',
    'lake_ice_depth',
    'lake_ice_temperature',
    'lake_mix_layer_depth',
    'lake_mix_layer_temperature',
    'lake_shape_factor',
    'lake_total_layer_temperature',
    'land_sea_mask',
    'large_scale_precipitation',
    'large_scale_precipitation_fraction',
    'large_scale_rain_rate',
    'large_scale_snowfall',
    'large_scale_snowfall_rate_water_equivalent',
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'low_cloud_cover',
    'low_vegetation_cover',
    'magnitude_of_turbulent_surface_stress',
    'maximum_2m_temperature_since_previous_post_processing',
    'maximum_individual_wave_height',
    'maximum_total_precipitation_rate_since_previous_post_processing',
    'mean_boundary_layer_dissipation',
    'mean_convective_precipitation_rate',
    'mean_convective_snowfall_rate',
    'mean_direction_of_total_swell',
    'mean_direction_of_wind_waves',
    'mean_eastward_gravity_wave_surface_stress',
    'mean_eastward_turbulent_surface_stress',
    'mean_evaporation_rate',
    'mean_gravity_wave_dissipation',
    'mean_large_scale_precipitation_fraction',
    'mean_large_scale_precipitation_rate',
    'mean_large_scale_snowfall_rate',
    'mean_magnitude_of_turbulent_surface_stress',
    'mean_northward_gravity_wave_surface_stress',
    'mean_northward_turbulent_surface_stress',
    'mean_period_of_total_swell',
    'mean_period_of_wind_waves',
    'mean_potential_evaporation_rate',
    'mean_runoff_rate',
    'mean_sea_level_pressure',
    'mean_snow_evaporation_rate',
    'mean_snowfall_rate',
    'mean_snowmelt_rate',
    'mean_square_slope_of_waves',
    'mean_sub_surface_runoff_rate',
    'mean_surface_direct_short_wave_radiation_flux',
    'mean_surface_direct_short_wave_radiation_flux_clear_sky',
    'mean_surface_downward_long_wave_radiation_flux',
    'mean_surface_downward_long_wave_radiation_flux_clear_sky',
    'mean_surface_downward_short_wave_radiation_flux',
    'mean_surface_downward_short_wave_radiation_flux_clear_sky',
    'mean_surface_downward_uv_radiation_flux',
    'mean_surface_latent_heat_flux',
    'mean_surface_net_long_wave_radiation_flux',
    'mean_surface_net_long_wave_radiation_flux_clear_sky',
    'mean_surface_net_short_wave_radiation_flux',
    'mean_surface_net_short_wave_radiation_flux_clear_sky',
    'mean_surface_runoff_rate',
    'mean_surface_sensible_heat_flux',
    'mean_top_downward_short_wave_radiation_flux',
    'mean_top_net_long_wave_radiation_flux',
    'mean_top_net_long_wave_radiation_flux_clear_sky',
    'mean_top_net_short_wave_radiation_flux',
    'mean_top_net_short_wave_radiation_flux_clear_sky',
    'mean_total_precipitation_rate',
    'mean_vertical_gradient_of_refractivity_inside_trapping_layer',
    'mean_vertically_integrated_moisture_divergence',
    'mean_wave_direction',
    'mean_wave_direction_of_first_swell_partition',
    'mean_wave_direction_of_second_swell_partition',
    'mean_wave_direction_of_third_swell_partition',
    'mean_wave_period',
    'mean_wave_period_based_on_first_moment',
    'mean_wave_period_based_on_first_moment_for_swell',
    'mean_wave_period_based_on_first_moment_for_wind_waves',
    'mean_wave_period_based_on_second_moment_for_swell',
    'mean_wave_period_based_on_second_moment_for_wind_waves',
    'mean_wave_period_of_first_swell_partition',
    'mean_wave_period_of_second_swell_partition',
    'mean_wave_period_of_third_swell_partition',
    'mean_zero_crossing_wave_period',
    'medium_cloud_cover',
    'minimum_2m_temperature_since_previous_post_processing',
    'minimum_total_precipitation_rate_since_previous_post_processing',
    'minimum_vertical_gradient_of_refractivity_inside_trapping_layer',
    'model_bathymetry',
    'near_ir_albedo_for_diffuse_radiation',
    'near_ir_albedo_for_direct_radiation',
    'normalized_energy_flux_into_ocean',
    'normalized_energy_flux_into_waves',
    'normalized_stress_into_ocean',
    'northward_gravity_wave_surface_stress',
    'northward_turbulent_surface_stress',
    'ocean_surface_stress_equivalent_10m_neutral_wind_direction',
    'ocean_surface_stress_equivalent_10m_neutral_wind_speed',
    'orography',  # deprecated
    'peak_wave_period',
    'period_corresponding_to_maximum_individual_wave_height',
    'potential_evaporation',
    'precipitation_type',
    'runoff',
    'sea_ice_cover',
    'sea_surface_temperature',
    'significant_height_of_combined_wind_waves_and_swell',
    'significant_height_of_total_swell',
    'significant_height_of_wind_waves',
    'significant_wave_height_of_first_swell_partition',
    'significant_wave_height_of_second_swell_partition',
    'significant_wave_height_of_third_swell_partition',
    'skin_reservoir_content',
    'skin_temperature',
    'slope_of_sub_gridscale_orography',
    'snow_albedo',
    'snow_density',
    'snow_depth',
    'snow_evaporation',
    'snowfall',
    'snowmelt',
    'soil_temperature_level_1',
    'soil_temperature_level_2',
    'soil_temperature_level_3',
    'soil_temperature_level_4',
    'soil_type',
    'standard_deviation_of_filtered_subgrid_orography',
    'standard_deviation_of_orography',
    'sub_surface_runoff',
    'surface_latent_heat_flux',
    'surface_net_solar_radiation',
    'surface_net_solar_radiation_clear_sky',
    'surface_net_thermal_radiation',
    'surface_net_thermal_radiation_clear_sky',
    'surface_pressure',
    'surface_runoff',
    'surface_sensible_heat_flux',
    'surface_solar_radiation_downward_clear_sky',
    'surface_solar_radiation_downwards',
    'surface_thermal_radiation_downward_clear_sky',
    'surface_thermal_radiation_downwards',
    'temperature_of_snow_layer',
    'toa_incident_solar_radiation',
    'top_net_solar_radiation',
    'top_net_solar_radiation_clear_sky',
    'top_net_thermal_radiation',
    'top_net_thermal_radiation_clear_sky',
    'total_cloud_cover',
    'total_column_cloud_ice_water',
    'total_column_cloud_liquid_water',
    'total_column_ozone',
    'total_column_rain_water',
    'total_column_snow_water',
    'total_column_supercooled_liquid_water',
    'total_column_water',
    'total_column_water_vapour',
    'total_precipitation',
    'total_sky_direct_solar_radiation_at_surface',
    'total_totals_index',
    'trapping_layer_base_height',
    'trapping_layer_top_height',
    'type_of_high_vegetation',
    'type_of_low_vegetation',
    'u_component_stokes_drift',
    'uv_visible_albedo_for_diffuse_radiation',
    'uv_visible_albedo_for_direct_radiation',
    'v_component_stokes_drift',
    'vertical_integral_of_divergence_of_cloud_frozen_water_flux',
    'vertical_integral_of_divergence_of_cloud_liquid_water_flux',
    'vertical_integral_of_divergence_of_geopotential_flux',
    'vertical_integral_of_divergence_of_kinetic_energy_flux',
    'vertical_integral_of_divergence_of_mass_flux',
    'vertical_integral_of_divergence_of_moisture_flux',
    'vertical_integral_of_divergence_of_ozone_flux',
    'vertical_integral_of_divergence_of_thermal_energy_flux',
    'vertical_integral_of_divergence_of_total_energy_flux',
    'vertical_integral_of_eastward_cloud_frozen_water_flux',
    'vertical_integral_of_eastward_cloud_liquid_water_flux',
    'vertical_integral_of_eastward_geopotential_flux',
    'vertical_integral_of_eastward_heat_flux',
    'vertical_integral_of_eastward_kinetic_energy_flux',
    'vertical_integral_of_eastward_mass_flux',
    'vertical_integral_of_eastward_ozone_flux',
    'vertical_integral_of_eastward_total_energy_flux',
    'vertical_integral_of_eastward_water_vapour_flux',
    'vertical_integral_of_energy_conversion',
    'vertical_integral_of_kinetic_energy',
    'vertical_integral_of_mass_of_atmosphere',
    'vertical_integral_of_mass_tendency',
    'vertical_integral_of_northward_cloud_frozen_water_flux',
    'vertical_integral_of_northward_cloud_liquid_water_flux',
    'vertical_integral_of_northward_geopotential_flux',
    'vertical_integral_of_northward_heat_flux',
    'vertical_integral_of_northward_kinetic_energy_flux',
    'vertical_integral_of_northward_mass_flux',
    'vertical_integral_of_northward_ozone_flux',
    'vertical_integral_of_northward_total_energy_flux',
    'vertical_integral_of_northward_water_vapour_flux',
    'vertical_integral_of_potential_and_internal_energy',
    'vertical_integral_of_potential_internal_and_latent_energy',
    'vertical_integral_of_temperature',
    'vertical_integral_of_thermal_energy',
    'vertical_integral_of_total_energy',
    'vertically_integrated_moisture_divergence',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'wave_spectral_directional_width',
    'wave_spectral_directional_width_for_swell',
    'wave_spectral_directional_width_for_wind_waves',
    'wave_spectral_kurtosis',
    'wave_spectral_peakedness',
    'wave_spectral_skewness',
    'zero_degree_level',
]

# ERA-5 pressure levels
PRESSURE_LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

# pressure level variables
PRESSURE_LEVEL_VARIABLES = [
    'divergence',
    'fraction_of_cloud_cover',
    'geopotential',
    'ozone_mass_mixing_ratio',
    'potential_vorticity',
    'relative_humidity',
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
    'specific_humidity',
    'specific_rain_water_content',
    'specific_snow_water_content',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'vorticity',
]


# ERA5-Land data variables
ERA5_LAND_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'evaporation_from_bare_soil',
    'evaporation_from_open_water_surfaces_excluding_oceans',
    'evaporation_from_the_top_of_canopy',
    'evaporation_from_vegetation_transpiration',
    'forecast_albedo',
    'lake_bottom_temperature',
    'lake_ice_depth',
    'lake_ice_temperature',
    'lake_mix_layer_depth',
    'lake_mix_layer_temperature',
    'lake_shape_factor',
    'lake_total_layer_temperature',
    'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation',
    'potential_evaporation',
    'runoff',
    'skin_reservoir_content',
    'skin_temperature',
    'snow_albedo',
    'snow_cover',
    'snow_density',
    'snow_depth',
    'snow_depth_water_equivalent',
    'snow_evaporation',
    'snowfall',
    'snowmelt',
    'soil_temperature_level_1',
    'soil_temperature_level_2',
    'soil_temperature_level_3',
    'soil_temperature_level_4',
    'sub_surface_runoff',
    'surface_latent_heat_flux',
    'surface_net_solar_radiation',
    'surface_net_thermal_radiation',
    'surface_pressure',
    'surface_runoff',
    'surface_sensible_heat_flux',
    'surface_solar_radiation_downwards',
    'surface_thermal_radiation_downwards',
    'temperature_of_snow_layer',
    'total_evaporation',
    'total_precipitation',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
]


# SINGLE_LEVEL_VARIABLES that are invalid for monthly requests
MISSING_MONTHLY_VARIABLES = [
    '10m_wind_gust_since_previous_post_processing',
    'maximum_2m_temperature_since_previous_post_processing',
    'minimum_2m_temperature_since_previous_post_processing',
    'maximum_total_precipitation_rate_since_previous_post_processing',
    'minimum_total_precipitation_rate_since_previous_post_processing',
]


# SINGLE_LEVEL_VARIABLES that are invalid for hourly requests
MISSING_HOURLY_VARIABLES = [
    '10m_wind_speed',
    'magnitude_of_turbulent_surface_stress',
    'mean_magnitude_of_turbulent_surface_stress',
]


def verify_dataset(
    dataset_name: str,
) -> None:

    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f'param:dataset_name must be in {DATASET_NAMES}!'
        )

# dataset_name -> map to a class -> that class will verify the variables


def list_variables(
    dataset_name: str,
    dataset_source: str = 'CDS',
) -> List[str]:
    """Returns a list of possible variables given a dataset_name / endpoint and data source"""
    # verify dataset name and source
    verify_dataset(dataset_name, dataset_source)

    # TODO: Abstract this, would get ugly if we added more datasets
    # NOTE: responsibility should be the class / dataset
    if dataset_source == 'AWS':
        if dataset_name != 'reanalysis-era5-single-levels':
            raise ValueError(
                f'param:dataset_source={dataset_source} only contains '
                f'dataset_name=reanalysis-era5-single-levels'
            )
        return list(AWS_VARIABLES_DICT.keys())

    if 'single-levels' in dataset_name:
        if 'monthly' in dataset_name:
            return [i for i in SINGLE_LEVEL_VARIABLES if i not in MISSING_MONTHLY_VARIABLES]
        else:
            return [i for i in SINGLE_LEVEL_VARIABLES if i not in MISSING_HOURLY_VARIABLES]
    elif 'pressure-levels' in dataset_name:
        return PRESSURE_LEVEL_VARIABLES
    elif 'land' in dataset_name:
        return ERA5_LAND_VARIABLES
    else:
        raise ValueError(f'Cannot return variables. Something went wrong.')
