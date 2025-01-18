import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from pySAMetrics import config


def diagnostic_fmse_z(
    fmse_array: np.array,
    z_array: np.array,
    data_array: np.array,
    time_step: int,
    nb_bins_fmse: int = 50,
    fmse_range: str = "max",
    bin_mode='sum'
):

    if type(data_array) not in [
        list,
        np.array,
        xr.core.dataarray.DataArray,
        np.ndarray,
    ]:
        raise ValueError(
            "data_array type is not standard, must be in [list, np.array, xarray.core.dataarray.DataArray, np.ndarray,]"
        )

    if type(data_array) not in [
        list,
        np.array,
        np.ndarray,
    ]:
        data_array = data_array.values

    if fmse_range not in ["max", "1-percentile"]:
        raise ValueError("fmse_range must be in [max, 1_percentile]")

    nz = z_array.shape[0]
    output_matrix = np.zeros((nz, nb_bins_fmse))

    data_array_i = data_array[time_step]
    fmse_array_i = fmse_array[time_step]

    if fmse_range == "max":
        total_min, total_max = (np.min(fmse_array_i), np.max(fmse_array_i))

    if fmse_range == "1-percentile":
        total_min, total_max = (np.percentile(fmse_array_i, 1), np.percentile(fmse_array_i, 99))

    total_range = np.linspace(config.FMSE_MIN, config.FMSE_MAX, nb_bins_fmse)

    fmse_array_i[fmse_array_i>config.FMSE_MAX]=config.FMSE_MAX
    fmse_array_i[fmse_array_i<config.FMSE_MIN]=config.FMSE_MIN




    for zz in range(nz - 1):

        ind_xy = np.array(
            [
                np.where(
                    np.logical_and(
                        total_range[i] <= fmse_array_i[zz],
                        fmse_array_i[zz] <= total_range[i + 1],
                    )
                )
                for i in range(total_range.shape[0] - 1)
            ],
            dtype="object",
        )

        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            #foo = np.nanmean(x, axis=1)
        
        value_zz_fmse=[]
        for i in range(len(ind_xy)):
            arr = data_array_i[zz, ind_xy[i][0], ind_xy[i][1]]
            if len(arr>0):
                #mean_value_zz_fmse.append(arr[np.argmax(np.absolute(arr))])
                if bin_mode=='sum':
                    value_zz_fmse.append(np.sum(arr))
                elif bin_mode=='mean':
                    value_zz_fmse.append(np.mean(arr))
                elif bin_mode=='max':
                    value_zz_fmse.append(arr[np.argmax(np.absolute(arr))])

            else:
                value_zz_fmse.append(np.nan)

        output_matrix[zz, 1:] = value_zz_fmse

    return output_matrix

###to select subdomain

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from pySAMetrics import config
from scipy.ndimage import generic_filter, uniform_filter
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallelization


def variance_filter(array, nx_sub, ny_sub):
    """Apply a variance filter over the image with a window size of (nx_sub, ny_sub)."""
    return generic_filter(array, np.var, size=(nx_sub, ny_sub))

def find_most_variable_subdomain_via_filter(array_2d, nx_sub, ny_sub, pad=10):
    """Find the subdomain of size (nx_sub, ny_sub) with the strongest variability using a filter."""
    # Apply padding to the array
    padded_array = np.pad(array_2d, pad_width=pad, mode='constant', constant_values=0)
    
    # Apply a variance filter over the padded array
    variability_map = variance_filter(padded_array, nx_sub, ny_sub)
    
    # Optionally, crop back to the original array size if needed (removing padding)
    cropped_variability_map = variability_map[pad:-pad, pad:-pad]
    
    # Find the coordinates of the maximum variability in the cropped map
    max_coords = np.unravel_index(np.argmax(cropped_variability_map), cropped_variability_map.shape)
    
    return max_coords, cropped_variability_map

def variance_filter_fast_with_padding(array, nx_sub, ny_sub, pad=1):
    """Apply a faster variance filter over the image using uniform_filter, with padding."""
    # Pad the array to simulate the behavior of the generic_filter pad
    padded_array = np.pad(array, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    # Compute the mean in the nx_sub by ny_sub window
    mean = uniform_filter(padded_array, size=(nx_sub, ny_sub))
    
    # Compute the mean of the squared values in the window
    mean_of_squares = uniform_filter(padded_array**2, size=(nx_sub, ny_sub))
    
    # Variance is E[x^2] - (E[x])^2
    variance = mean_of_squares - mean**2
    
    # Return the valid part of the result (exclude padded areas)
    return variance[pad:-pad, pad:-pad]

def find_most_variable_subdomain_via_filter_fast(array_2d, nx_sub, ny_sub, pad=1):
    """Find the subdomain with the strongest variability using a fast filter."""
    # Apply a fast variance filter over the array
    variability_map = variance_filter_fast_with_padding(array_2d, nx_sub, ny_sub, pad=pad)
    
    # Find the coordinates of the maximum variability
    max_coords = np.unravel_index(np.argmax(variability_map), variability_map.shape)
    
    return max_coords, variability_map

def subdomain_indices(x_center, y_center, nx_sub, ny_sub, nx, ny):
        
        # Calculate the subdomain indices considering periodic boundaries
        x_indices = (np.arange(x_center - nx_sub // 2, x_center + nx_sub // 2) % nx)
        y_indices = (np.arange(y_center - ny_sub // 2, y_center + ny_sub // 2) % ny)
        
        return x_indices, y_indices

def extract_periodic_subdomain(array,x_indices,y_indices ):
    """
    Extracts a subdomain from a doubly periodic domain around a given center.
    
    Parameters:
    - array: 2D array (image) from which to extract the subdomain.
    - x_center: The x-coordinate of the center of the subdomain.
    - y_center: The y-coordinate of the center of the subdomain.
    - nx_sub: The width of the subdomain.
    - ny_sub: The height of the subdomain.

    Returns:
    - subdomain: The extracted subdomain as a 2D array.
    """
    
    # Extract the subdomain
    subdomain = array[np.ix_(x_indices, y_indices)]
    
    return subdomain

