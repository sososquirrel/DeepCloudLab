"""
Module for computing basic variables from simulation datasets.
"""

# Standard Library Imports
import logging
from typing import Optional

# Third-Party Imports
import numpy as np
import xarray as xr

# pySAMetrics Imports
import pySAMetrics
from pySAMetrics.relative_humidity import get_qsatt
from pySAMetrics.condensation_rate import get_condensation_rate
from pySAMetrics.utils import expand_array_to_tzyx_array
from pySAMetrics.phase_split import split_QN, split_QP

# Module logging
logger = logging.getLogger(__name__)

def set_basic_variables_from_dataset_add(simulation)-> None:
    QN = simulation.dataset_3d.QN.values
    QP = simulation.dataset_3d.QP.values
    TABS = simulation.dataset_3d.TABS.values

    QN_ice, QN_liq = split_QN(QN=QN, TABS=TABS, t0=273.15)
    QP_ice, QP_liq = split_QP(QP=QP, TABS=TABS, t0=273.15)

    Lf = 3.34e5

    FFMSE = simulation.dataset_computed_3d.FMSE.values - Lf*QN_ice/1000

    simulation.dataset_computed_3d = simulation.dataset_computed_3d.assign({
            'QN_ice': (('time', 'z', 'y', 'x'), QN_ice, {
                'long_name': 'Non-Precipitating Water (Snow)',
                'units': 'g/kg'
            }),
            'QN_liq': (('time', 'z', 'y', 'x'), QN_liq, {
                'long_name': 'Non-Precipitating Water (Rain)',
                'units': 'g/kg'
            }),
            'QP_ice': (('time', 'z', 'y', 'x'), QP_ice, {
                'long_name': 'Precipitating Water (Snow)',
                'units': 'g/kg'
            }),
            'QP_liq': (('time', 'z', 'y', 'x'), QP_liq, {
                'long_name': 'Precipitating Water (Rain)',
                'units': 'g/kg'
            }),  # <--- ADD THIS COMMA
            'FFMSE': (('time', 'z', 'y', 'x'), FFMSE, {
                'long_name': 'real Free Moist Static Energy',
                'units': 'J kg^-1'
            })
        })


def set_basic_variables_from_dataset(simulation) -> None:
    """
    Compute basic variables from dataset variables, adapted to (nt,nz,ny,nx) data.
    
    Args:
        simulation: Simulation object containing the necessary datasets
    """
    logger.info("Computing basic variables from dataset")
    
    # Cache values to avoid repeated access
    tabs_values = simulation.dataset_3d.TABS.values
    qv_values = simulation.dataset_3d.QV.values / 1000  # in kg/kg
    p_values = simulation.dataset_1d.p.values[:simulation.nz]

    # Height array expanded to match the shape
    z_3d_in_time = expand_array_to_tzyx_array(
        time_dependence=False,
        input_array=simulation.Z,
        final_shape=np.array((simulation.nt, simulation.nz, simulation.ny, simulation.nx)),
    )

    # Pressure array expanded to match the shape
    pressure_3d_in_time = expand_array_to_tzyx_array(
        time_dependence=False,
        input_array=p_values,
        final_shape=np.array((simulation.nt, simulation.nz, simulation.ny, simulation.nx)),
    )

    # Compute Free Moist Static Energy (FMSE)
    FMSE = (
        pySAMetrics.HEAT_CAPACITY_AIR * tabs_values
        + pySAMetrics.GRAVITY * z_3d_in_time
        + pySAMetrics.LATENT_HEAT_AIR * qv_values
    )

    # Compute Virtual Temperature
    VIRTUAL_TEMPERATURE = tabs_values * (
        1 + (pySAMetrics.MIXING_RATIO_AIR_WATER_VAPOR - 1)
        / pySAMetrics.MIXING_RATIO_AIR_WATER_VAPOR * qv_values
    )

    # Compute vertical mean of virtual temperature
    vertical_mean_virtual_temperature_3d_in_time = expand_array_to_tzyx_array(
        time_dependence=True,
        input_array=np.mean(VIRTUAL_TEMPERATURE, axis=(2, 3)),
        final_shape=np.array((simulation.nt, simulation.nz, simulation.ny, simulation.nx)),
    )

    # Compute Potential Temperature
    POTENTIAL_TEMPERATURE = (
        tabs_values
        * (pySAMetrics.STANDARD_REFERENCE_PRESSURE / pressure_3d_in_time)
        ** pySAMetrics.GAS_CONSTANT_OVER_HEAT_CAPACITY_AIR
    )

    # Compute Buoyancy
    BUOYANCY = (
        pySAMetrics.GRAVITY
        * (VIRTUAL_TEMPERATURE - vertical_mean_virtual_temperature_3d_in_time)
        / vertical_mean_virtual_temperature_3d_in_time
    )

    # Compute Vorticity
    VORTICITY = np.gradient(simulation.dataset_3d.U.values, simulation.Z, axis=1) - np.gradient(
        simulation.dataset_3d.W.values, simulation.X, axis=3
    )

    # Compute saturation specific humidity
    QSATT = get_qsatt(p_values, tabs_values)

    # Compute Relative Humidity
    RH = qv_values / QSATT

    # Compute Condensation Rate
    CR = get_condensation_rate(
        vertical_velocity=simulation.dataset_3d.W.values,
        density=simulation.dataset_1d.RHO.values,
        humidity=qv_values,
    )
    
    # Compute vertical mass flux
    RHO_W = expand_array_to_tzyx_array(
        input_array=simulation.dataset_1d.RHO.values[-simulation.nt:],
        final_shape=(simulation.nt, simulation.nz, simulation.nx, simulation.ny),
        time_dependence=True
    ) * simulation.dataset_3d.W.values
    
    # Create 3D computed dataset
    simulation.dataset_computed_3d = xr.Dataset({
        'FMSE': (('time', 'z', 'y', 'x'), FMSE, {
            'long_name': 'Free Moist Static Energy',
            'units': 'J kg^-1'
        }),
        'VIRTUAL_TEMPERATURE': (('time', 'z', 'y', 'x'), VIRTUAL_TEMPERATURE, {
            'long_name': 'Virtual Temperature',
            'units': 'K'
        }),
        'POTENTIAL_TEMPERATURE': (('time', 'z', 'y', 'x'), POTENTIAL_TEMPERATURE, {
            'long_name': 'Potential Temperature',
            'units': 'K'
        }),
        'BUOYANCY': (('time', 'z', 'y', 'x'), BUOYANCY, {
            'long_name': 'Buoyancy',
            'units': 'm s^-2'
        }),
        'VORTICITY': (('time', 'z', 'y', 'x'), VORTICITY, {
            'long_name': 'Vorticity',
            'units': 's^-1'
        }),
        'RHO_W': (('time', 'z', 'y', 'x'), RHO_W, {
            'long_name': 'Vertical Mass Flux',
            'units': 'kg.m/s'
        })
    })

    # Create 2D computed dataset
    simulation.dataset_computed_2d = xr.Dataset({
        'CR': (('time', 'y', 'x'), CR, {
            'long_name': 'Condensation rate',
            'units': 'mm'
        })
    })
    
    logger.info("Basic variables computed and stored in dataset_computed_3d and dataset_computed_2d")


def set_3d_condensation_rate(simulation) -> None:
    """
    Compute 3D condensation rate.
    
    Args:
        simulation: Simulation object containing the necessary datasets
    """
    logger.info("Computing 3D condensation rate")
    
    # Ensure W is available
    if not hasattr(simulation, 'W') or simulation.W is None:
        simulation.W = simulation.dataset_3d.W.values
    
    # Compute 3D condensation rate
    simulation.CR_3D = get_condensation_rate(
        vertical_velocity=simulation.W.values if hasattr(simulation.W, 'values') else simulation.W,
        density=simulation.dataset_1d.RHO.values,
        humidity=simulation.dataset_3d.QV.values,
        return_3D=True,
    )
    
    logger.info("3D condensation rate computed and stored in CR_3D attribute")