"""
Example script demonstrating how to use the entrainment calculation functions.

This script shows how to calculate E (entrainment), D (detrainment), and E-D 
for a given simulation using the functions from diagnostic_fmse_v2.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the pySAMetrics path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths
from pySAMetrics.diagnostic_fmse_v2 import calculate_entrainment_detrainment, calculate_entrainment_detrainment_timeseries

def load_simulation(simu_parameters, path_raw_data='/Volumes/LaCie/000_POSTDOC_2025/long_high_res'):
    """
    Load and run the simulation for the given parameters.
    """
    try:
        paths = generate_simulation_paths(**simu_parameters, folder_path=path_raw_data)
        print(f"Loading simulation with paths: {paths}")
        
        simu = Simulation(data_folder_paths=[paths['path_1d'], paths['path_2d'], paths['path_3d']],
                          **simu_parameters)
        return simu
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the simulation paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
    
    return None

def example_single_time():
    """
    Example: Calculate E, D, and E-D for a single time index
    """
    print("=== Example: Single Time Calculation ===")
    
    # Define simulation parameters (adjust as needed)
    simu_parameters = {
        'velocity': '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': '4'
    }
    
    # Load simulation
    simu = load_simulation(simu_parameters)
    if simu is None:
        print("Failed to load simulation. Please check your data paths.")
        return
    
    # Load the simulation data
    simu.load(backup_folder_path='/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')
    
    # Calculate E, D, and E-D for time index 60
    time_index = 60
    result = calculate_entrainment_detrainment(simu, time_index, epsilon=1.0)
    
    print(f"Results for time index {time_index}:")
    print(f"E shape: {result['E'].shape}")
    print(f"D shape: {result['D'].shape}")
    print(f"E-D shape: {result['E_minus_D'].shape}")
    print(f"Height levels: {len(result['z'])}")
    
    # Plot the results
    z = result['z'] / 1000  # Convert to km
    
    plt.figure(figsize=(15, 5))
    
    # Plot E
    plt.subplot(1, 3, 1)
    plt.plot(result['E'], z, 'b-', linewidth=2)
    plt.xlabel('E (1/m)')
    plt.ylabel('Height (km)')
    plt.title('Entrainment Rate (E)')
    plt.grid(True, alpha=0.3)
    
    # Plot D
    plt.subplot(1, 3, 2)
    plt.plot(result['D'], z, 'r-', linewidth=2)
    plt.xlabel('D (1/m)')
    plt.ylabel('Height (km)')
    plt.title('Detrainment Rate (D)')
    plt.grid(True, alpha=0.3)
    
    # Plot E-D
    plt.subplot(1, 3, 3)
    plt.plot(result['E_minus_D'], z, 'g-', linewidth=2)
    plt.xlabel('E-D (1/m)')
    plt.ylabel('Height (km)')
    plt.title('E - D')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def example_timeseries():
    """
    Example: Calculate E, D, and E-D for multiple time indices
    """
    print("=== Example: Time Series Calculation ===")
    
    # Define simulation parameters (adjust as needed)
    simu_parameters = {
        'velocity': '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': '4'
    }
    
    # Load simulation
    simu = load_simulation(simu_parameters)
    if simu is None:
        print("Failed to load simulation. Please check your data paths.")
        return
    
    # Load the simulation data
    simu.load(backup_folder_path='/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')
    
    # Calculate E, D, and E-D for time indices 50-70
    time_indices = list(range(50, 71))
    result = calculate_entrainment_detrainment_timeseries(simu, time_indices, epsilon=1.0)
    
    print(f"Results for time indices {time_indices[0]}-{time_indices[-1]}:")
    print(f"E shape: {result['E'].shape}")
    print(f"D shape: {result['D'].shape}")
    print(f"E-D shape: {result['E_minus_D'].shape}")
    
    # Plot time-height cross sections
    z = result['z'] / 1000  # Convert to km
    time_indices_plot = result['time_indices']
    
    plt.figure(figsize=(15, 5))
    
    # Plot E
    plt.subplot(1, 3, 1)
    plt.contourf(time_indices_plot, z, result['E'].T, levels=20, cmap='RdBu_r')
    plt.colorbar(label='E (1/m)')
    plt.xlabel('Time Index')
    plt.ylabel('Height (km)')
    plt.title('Entrainment Rate (E)')
    
    # Plot D
    plt.subplot(1, 3, 2)
    plt.contourf(time_indices_plot, z, result['D'].T, levels=20, cmap='RdBu_r')
    plt.colorbar(label='D (1/m)')
    plt.xlabel('Time Index')
    plt.ylabel('Height (km)')
    plt.title('Detrainment Rate (D)')
    
    # Plot E-D
    plt.subplot(1, 3, 3)
    plt.contourf(time_indices_plot, z, result['E_minus_D'].T, levels=20, cmap='RdBu_r')
    plt.colorbar(label='E-D (1/m)')
    plt.xlabel('Time Index')
    plt.ylabel('Height (km)')
    plt.title('E - D')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run examples
    example_single_time()
    example_timeseries() 