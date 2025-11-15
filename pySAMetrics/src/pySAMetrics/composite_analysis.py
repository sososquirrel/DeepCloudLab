"""
Module for composite analysis of simulation data.
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
from pySAMetrics.utils import make_parallel
from pySAMetrics.composite import instant_mean_extraction_data_over_extreme

# Module logging
logger = logging.getLogger(__name__)


def set_composite_variables(
    simulation,
    data_name: str,
    variable_to_look_for_extreme: str,
    extreme_events_choice: str,
    x_margin: int,
    y_margin: int,
    parallelize: bool = False,
    return_3D: bool = False,
    dataset_for_variable_2d: str = "dataset_2d",
    dataset_for_variable_3d: str = "dataset_3d",
    return_1D: bool = False,
    mask_mode: bool = False,
    var_mask_name: str = '',
    dataset_mask_3d: str = "dataset_3d",
) -> None:
    """
    Compute the composite (conditional mean) of 2D or 3D variables evolving in time.
    
    Args:
        simulation: Simulation object containing the datasets
        data_name: Name of the variable composite method is applying to
        variable_to_look_for_extreme: Name of the variable that describes extreme events
        extreme_events_choice: max 1-percentile or 10-percentile
        x_margin: Width of window zoom
        y_margin: Depth of window zoom
        parallelize: Use all CPU power if True
        return_3D: Return 3D data if True
        dataset_for_variable_2d: Dataset containing the 2D variable
        dataset_for_variable_3d: Dataset containing the 3D variable
        return_1D: Return 1D data if True
        mask_mode: Apply mask if True
        var_mask_name: Name of the mask variable
        dataset_mask_3d: Dataset containing the mask
    """
    logger.info(f"Computing composite variables for {data_name}")
    
    # Get 3D data
    if dataset_for_variable_3d == "":
        data_3d = getattr(simulation, data_name)
    else:
        data_3d = getattr(getattr(simulation, dataset_for_variable_3d), data_name)
            
    # Get 2D data
    if dataset_for_variable_2d == "":
        data_2d = getattr(simulation, variable_to_look_for_extreme)
    else:
        data_2d = getattr(
            getattr(simulation, dataset_for_variable_2d), variable_to_look_for_extreme
        )

    # Convert to numpy arrays if necessary
    if isinstance(data_2d, xr.core.dataarray.DataArray):
        data_2d = data_2d.values

    if isinstance(data_3d, xr.core.dataarray.DataArray):
        data_3d = data_3d.values

    # Apply mask if in mask mode
    if mask_mode:
        data_mask = getattr(getattr(simulation, dataset_mask_3d), var_mask_name)
        if isinstance(data_mask, xr.core.dataarray.DataArray):
            data_mask = data_mask.values
        data_3d = data_3d * data_mask

    # Ensure data is float32
    data_2d = np.array(data_2d, dtype=np.float32)
    data_3d = np.array(data_3d, dtype=np.float32)


    # Compute composites
    if parallelize:
        logger.info(f"Using parallel processing with {pySAMetrics.N_CPU} CPUs")
        parallel_composite = make_parallel(
            function=instant_mean_extraction_data_over_extreme, 
            nprocesses=pySAMetrics.N_CPU
        )
        composite_variable = parallel_composite(
            iterable_values_1=data_3d,
            iterable_values_2=data_2d,
            extreme_events_choice=extreme_events_choice,
            x_margin=x_margin,
            y_margin=y_margin,
            return_3D=return_3D,
        )
    else:
        logger.info("Computing composites sequentially")
        composite_variable = []
        data = data_3d
        for image, variable_extreme in tqdm(zip(data, data_2d), total=len(data)):


            composite_variable.append(
                instant_mean_extraction_data_over_extreme(
                    data=image,
                    variable_to_look_for_extreme=variable_extreme,
                    extreme_events_choice=extreme_events_choice,
                    x_margin=x_margin,
                    y_margin=y_margin,
                    return_3D=return_3D,
                )

            )

    # Average over time
    composite_variable = np.array(composite_variable)
    composite_variable = np.mean(composite_variable, axis=0)

    # Set attribute with appropriate name
    if return_3D:
        attr_name = f"{data_name}_composite_{variable_to_look_for_extreme}_3D"
    elif mask_mode:
        attr_name = f"{data_name}_composite_{variable_to_look_for_extreme}_mask_{var_mask_name}"
    else:
        attr_name = f"{data_name}_composite_{variable_to_look_for_extreme}"
    
    setattr(simulation, attr_name, composite_variable)
    
    # Set 1D attribute if requested
    if return_1D:
        if return_3D:
            setattr(
                simulation,
                f"{data_name}_composite_{variable_to_look_for_extreme}_1D",
                composite_variable[:, x_margin, y_margin],
            )
        else:
            setattr(
                simulation,
                f"{data_name}_composite_{variable_to_look_for_extreme}_1D",
                composite_variable[:, x_margin],
            )
            
    logger.info(f"Composite variables for {data_name} computed successfully")
