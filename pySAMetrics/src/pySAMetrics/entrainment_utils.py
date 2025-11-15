from sympy import diff
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pySAMetrics.config as config

V_T=0.4 #m/s
L_f = 334e5

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
    fmse = simu.dataset_computed_3d.FFMSE.values[time_index]   # shape: (nz, nx, ny) #FFMSE is the real fmse
    z = simu.dataset_3d.z.values  # height levels
    rho_3d = simu.dataset_1d.RHO[time_index].values[:, np.newaxis, np.newaxis]

    qn=simu.dataset_computed_3d.QP_ice.values[time_index]/1000
    L_f_Pice = L_f*V_T*rho_3d*np.gradient(qn,z,axis=0)
    L_f_Pice = np.mean(L_f_Pice, axis=(-1, -2))
    
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
        numerator=np.where(mean_rho_w_plus != 0,
                           L_f_Pice / mean_rho_w_plus,
                           0)

        numerator = numerator-np.gradient(mean_fmse_plus, z)
        #numerator = -np.gradient(mean_fmse_plus, z)

        safe_div = np.where(diff_h != 0,
                           numerator / diff_h,
                           0)
        E = safe_div
    
    
    # Calculate detrainment rate D
    with np.errstate(divide='ignore', invalid='ignore'):
        # D = E - (1/M+) * dM+/dz
        numerator = np.gradient(mean_rho_w_plus,z)
        mass_correction = np.where(mean_rho_w_plus != 0,
                                  numerator / mean_rho_w_plus ,0)
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
        'mean_rho_w_plus': mean_rho_w_plus,
        'lf_pice' : L_f_Pice,
        'diff_h' : diff_h
    }