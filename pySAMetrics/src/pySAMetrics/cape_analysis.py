"""
Module for computing CAPE (Convective Available Potential Energy) in simulations.
"""

# Standard Library Imports
import logging
from typing import Optional, Tuple, List, Union

# Third-Party Imports
import numpy as np
import xarray as xr
from tqdm import tqdm

# pySAMetrics Imports
import pySAMetrics
from pySAMetrics.utils import make_parallel, max_point_wise
from pySAMetrics.cape.cape_functions import get_parcel_ascent

# Module logging
logger = logging.getLogger(__name__)


def get_cape(
    simulation,
    temperature: str = "TABS",
    vertical_array: str = "z",
    pressure: str = "p",
    humidity_ground: str = "QV",
    parallelize: bool = True,
    set_parcel_ascent_composite_1d: bool = False,
) -> None:
    """
    Calculate Convective Available Potential Energy (CAPE) for the simulation.
    
    Args:
        simulation: Simulation object containing the datasets
        temperature: Name of the temperature variable
        vertical_array: Name of the vertical coordinate variable
        pressure: Name of the pressure variable
        humidity_ground: Name of the humidity variable
        parallelize: Use parallel processing if True
        set_parcel_ascent_composite_1d: Set parcel ascent composite if True
    """
    logger.info("Computing CAPE")
    
    # Get the variables as numpy arrays
    temperature_array = getattr(simulation.dataset_3d, temperature).values
    nt, nz, ny, nx = temperature_array.shape

    z_array = getattr(simulation.dataset_3d, vertical_array).values
    pressure_array = getattr(simulation.dataset_1d, pressure).values[:nz]
    humidity_ground_array = getattr(simulation.dataset_3d, humidity_ground).values[:, 0, :, :]

    # Calculate parcel ascent
    if parallelize:
        logger.info(f"Computing parcel ascent in parallel using {pySAMetrics.N_CPU} CPUs")
        parallel_parcel_ascent = make_parallel(
            function=get_parcel_ascent, 
            nprocesses=pySAMetrics.N_CPU
        )
        parcel_ascent = parallel_parcel_ascent(
            iterable_values_1=temperature_array,
            iterable_values_2=humidity_ground_array / 1000,  # Convert to kg/kg
            pressure=pressure_array,
            vertical_array=z_array,
        )
    else:
        logger.info("Computing parcel ascent sequentially")
        parcel_ascent = []
        for temperature_i, humidity_ground_i in tqdm(zip(
            temperature_array,
            humidity_ground_array / 1000,  # Convert to kg/kg
        ), total=len(temperature_array)):
            T_parcel_i = get_parcel_ascent(
                temperature=temperature_i,
                humidity_ground=humidity_ground_i,
                pressure=pressure_array,
                vertical_array=z_array,
            )
            parcel_ascent.append(T_parcel_i)

    parcel_ascent = np.array(parcel_ascent)

    # Calculate CAPE
    dz = np.gradient(z_array)
    dz_3d = pySAMetrics.utils.expand_array_to_tzyx_array(
        input_array=dz, 
        time_dependence=False, 
        final_shape=temperature_array.shape
    )

    cape = pySAMetrics.GRAVITY * np.sum(
        dz_3d * max_point_wise(
            matrix_1=np.zeros_like(parcel_ascent),
            matrix_2=((parcel_ascent - temperature_array) / temperature_array),
        ),
        axis=1,
    )

    simulation.dataset_computed_2d = simulation.dataset_computed_2d.assign(
                CAPE=(('time','y', 'x'), cape, {
                    'long_name': 'CAPE',
                    'units': 'J/kg/m2'
                })
            )
                    
    #simulation.dataset_computed_3d = simulation.dataset_computed_3d.assign(
    #            PARCEL_ASCENT=(('time','z', 'y','x'), parcel_ascent, {
    #                'long_name': 'Parcel ascent',
    #                'units': 'J/kg'
    #            })
    #        )