from pySAMetrics import config
import numpy as np
from tqdm import tqdm
import xarray as xr

total_range = np.linspace(config.FMSE_MIN, config.FMSE_MAX, 50)


def get_isentropic_dataset(simulation):
    mass_flux_sum = get_isentropic_var(simulation, mode='sum')
    mass_flux_mean = get_isentropic_var(simulation, mode='mean')

    # Create the final xarray Dataset
    simulation.dataset_isentropic = xr.Dataset({
        'RHO_W_sum': (('time', 'z', 'fmse'), mass_flux_sum, {
            'long_name': 'Vertical Mass Flux',
            'units': 'kg/m2.s'
        }),
        'RHO_W_mean': (('time', 'z', 'fmse'), mass_flux_mean, {
            'long_name': 'Vertical Mass Flux',
            'units':'kg/m2.s'
        }),})

def get_isentropic_var(simu, total_range=total_range, mode='sum', mode_rhow=True, other_var=None):
    if mode_rhow:
        entropy = simu.dataset_computed_3d.FMSE.values
        mass_flux = simu.dataset_computed_3d.RHO_W.values
    else:
        if other_var is None:
            raise ValueError("When mode_rhow is False, 'other_var' must be provided.")
        
        try:
            entropy = simu.dataset_computed_3d.FMSE.values
        except AttributeError:
            raise AttributeError("Could not retrieve FMSE from 'simu.dataset_computed_3d'. Make sure it exists.")
        
        if other_var.shape != entropy.shape:
            raise ValueError(
                f"'other_var' must have the same shape as 'entropy'. "
                f"Got other_var.shape = {other_var.shape}, entropy.shape = {entropy.shape}."
            )
        mass_flux = other_var

    nt, nz, _ , _ = mass_flux.shape
    ns = len(total_range)


    shape = entropy.shape
    entropy =entropy.reshape((shape[0]*shape[1], -1))

    #histogram_tensor = np.stack([np.histogram(tensor_slice, bins=total_range)[0] for tensor_slice in tqdm(entropy)])
    #histogram_tensor = histogram_tensor.reshape((480, 64, histogram_tensor.shape[-1]))

    mass_flux = mass_flux.reshape((shape[0]*shape[1], -1))

    indices = np.stack([np.digitize(tensor_slice, bins=total_range) for tensor_slice in tqdm(entropy)])
    indices = indices - 1 #digitize from 1 to N

    indices = [[np.where(xx==bin)[0] for bin in range(ns)]for xx in indices]
    indices_reshaped = [indices[i*nz:(i+1)*nz] for i in range(nt)]

    count=0
    output = np.zeros((nt, nz, ns))
    for tt in tqdm(range(nt)):
        for zz in range(nz):
            aa = mass_flux[tt*nz + zz]
            if mode=='sum':
                output[tt, zz] = [np.sum(aa[indices_reshaped[tt][zz][i]]) for i in range(ns)]
            if mode=='mean':
                output[tt, zz] = [np.mean(aa[indices_reshaped[tt][zz][i]]) for i in range(ns)]
            #if mode=='max':
                #output[tt, zz] = [np.max(aa[indices_reshaped[tt][zz][i]]) for i in range(ns)]
            #if mode=='min':
                #output[tt, zz] = [np.max(aa[indices_reshaped[tt][zz][i]]) for i in range(ns)]
    return output


def get_isentropic_counts(simu, total_range=total_range):
    """
    Returns the number of grid points per isentropic bin at each time and vertical level.
    
    Parameters:
        simu: object with `dataset_computed_3d.FMSE.values` of shape (nt, nz, nx, ny)
        total_range: list or array of bin edges for FMSE (isentropic bins)

    Returns:
        output: np.ndarray of shape (nt, nz, nbins), number of grid points per bin
    """
    entropy = simu.dataset_computed_3d.FMSE.values  # shape: (nt, nz, nx, ny)
    nt, nz, nx, ny = entropy.shape
    ns = len(total_range)  # number of bins

    entropy = entropy.reshape((nt * nz, -1))  # shape: (nt * nz, nx * ny)

    indices = np.stack([np.digitize(tensor_slice, bins=total_range) for tensor_slice in tqdm(entropy)])
    indices = indices - 1 #digitize from 1 to N

    indices = [[np.where(xx==bin)[0] for bin in range(ns)]for xx in indices]
    indices_reshaped = [indices[i*nz:(i+1)*nz] for i in range(nt)]

    output = np.zeros((nt, nz, ns))
    for tt in tqdm(range(nt)):
        for zz in range(nz):
            output[tt, zz] = [np.log(1+len(indices_reshaped[tt][zz][i])) for i in range(ns)]
    return output

def add_counts_to_isentropic_dataset(simulation):
    counts_isentropic = get_isentropic_counts(simulation, total_range)
    # Check if self.dataset_2d exists
    if hasattr(simulation, 'dataset_isentropic') and simulation.dataset_isentropic is not None:
        # If self.dataset_isentropic exists, assign the new variables
        simulation.dataset_isentropic = simulation.dataset_isentropic.assign(
            COUNTS=(('time', 'z', 'fmse'), counts_isentropic, {
            'long_name': 'Number/bins',
            'units': '1'
                })
            )

def calculate_entrainment_detrainment(simu, time_index, epsilon=1.0):
    """
    Calculate entrainment (E), detrainment (D), and their difference (E-D) for a given simulation.
    
    Parameters:
    -----------
    simu : Simulation object
        The simulation object containing 3D computed data
    time_index : int
        Time index for which to calculate E and D
    epsilon : float, optional
        Threshold for separating positive and negative vertical velocity regions (default: 1.0)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'E': entrainment rate (1/m)
        - 'D': detrainment rate (1/m) 
        - 'E_minus_D': difference between entrainment and detrainment (1/m)
        - 'z': height levels (m)
        - 'time_index': the time index used
    """
    import numpy as np
    
    # Get the data
    rho_w = simu.dataset_computed_3d.RHO_W.values[time_index]  # shape: (nz, nx, ny)
    fmse = simu.dataset_computed_3d.FMSE.values[time_index]   # shape: (nz, nx, ny)
    z = simu.dataset_3d.z.values  # height levels
    
    # Separate positive and negative vertical velocity regions
    rho_w_plus = np.full_like(rho_w, np.nan)
    rho_w_minus = np.full_like(rho_w, np.nan)
    fmse_plus = np.full_like(fmse, np.nan)
    fmse_minus = np.full_like(fmse, np.nan)
    
    # Apply threshold
    rho_w_plus[rho_w > epsilon] = rho_w[rho_w > epsilon]
    rho_w_minus[rho_w < epsilon] = rho_w[rho_w < epsilon]
    fmse_plus[rho_w > epsilon] = fmse[rho_w > epsilon]
    fmse_minus[rho_w < epsilon] = fmse[rho_w < epsilon]
    
    # Calculate horizontal means
    mean_fmse_plus = np.nanmean(fmse_plus, axis=(-1, -2))  # shape: (nz,)
    mean_fmse = np.nanmean(fmse, axis=(-1, -2))           # shape: (nz,)
    mean_rho_w_plus = np.nanmean(rho_w_plus, axis=(-1, -2))  # shape: (nz,)
    
    # Calculate difference in FMSE
    diff_h = mean_fmse_plus - mean_fmse
    
    # Calculate entrainment rate E
    with np.errstate(divide='ignore', invalid='ignore'):
        # E = -dH+/dz / (H+ - H_bar)
        safe_div = np.where(mean_fmse_plus != 0,
                           np.gradient(mean_fmse_plus, z) / diff_h,
                           0)
        E = -safe_div
    
    # Calculate detrainment rate D
    with np.errstate(divide='ignore', invalid='ignore'):
        # D = E - (1/M+) * dM+/dz
        mass_correction = np.where(mean_rho_w_plus != 0,
                                  1 / mean_rho_w_plus * np.gradient(mean_rho_w_plus, z),
                                  0)
        D = E - mass_correction
    
    # Calculate E - D
    E_minus_D = E - D
    
    return {
        'E': E,
        'D': D, 
        'E_minus_D': E_minus_D,
        'z': z,
        'time_index': time_index,
        'mean_fmse_plus': mean_fmse_plus,
        'mean_fmse': mean_fmse,
        'mean_rho_w_plus': mean_rho_w_plus
    }

def calculate_entrainment_detrainment_timeseries(simu, time_indices=None, epsilon=1.0):
    """
    Calculate entrainment (E), detrainment (D), and their difference (E-D) for multiple time indices.
    
    Parameters:
    -----------
    simu : Simulation object
        The simulation object containing 3D computed data
    time_indices : list or array, optional
        List of time indices to calculate E and D for. If None, calculates for all available times.
    epsilon : float, optional
        Threshold for separating positive and negative vertical velocity regions (default: 1.0)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'E': entrainment rate array of shape (nt, nz) (1/m)
        - 'D': detrainment rate array of shape (nt, nz) (1/m) 
        - 'E_minus_D': difference array of shape (nt, nz) (1/m)
        - 'z': height levels (m)
        - 'time_indices': the time indices used
    """
    import numpy as np
    
    # Get the data dimensions
    nt_total, nz, nx, ny = simu.dataset_computed_3d.RHO_W.values.shape
    
    # If no time indices specified, use all available times
    if time_indices is None:
        time_indices = list(range(nt_total))
    
    # Initialize output arrays
    E_array = np.full((len(time_indices), nz), np.nan)
    D_array = np.full((len(time_indices), nz), np.nan)
    E_minus_D_array = np.full((len(time_indices), nz), np.nan)
    
    # Get height levels
    z = simu.dataset_3d.z.values
    
    # Calculate for each time index
    for i, t_idx in enumerate(time_indices):
        if t_idx >= nt_total:
            print(f"Warning: Time index {t_idx} exceeds available time range (0-{nt_total-1}). Skipping.")
            continue
            
        # Get single time result
        result = calculate_entrainment_detrainment(simu, t_idx, epsilon)
        
        # Store in arrays
        E_array[i] = result['E']
        D_array[i] = result['D']
        E_minus_D_array[i] = result['E_minus_D']
    
    return {
        'E': E_array,
        'D': D_array,
        'E_minus_D': E_minus_D_array,
        'z': z,
        'time_indices': time_indices
    }