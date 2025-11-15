import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Ensure pySAMetrics is available in your Python environment
import pySAMetrics
from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths


def load_simulation(simu_parameters, i=1000, path_raw_data='/Volumes/LaCie/000_POSTDOC_2025/long_high_res'):
    """
    Load and run the simulation for the given parameters.

    Parameters:
    - simu_parameters (dict): Dictionary containing simulation parameters.

    Returns:
    - Simulation object or None if failed to load.
    """
    try:
        if i != 1000:
            paths = {
                'path_3d': os.path.join(path_raw_data, f'3D/split_{i+1}.nc'),
                'path_2d': os.path.join(path_raw_data, f'2D/split_{i+1}.nc'),
                'path_1d': os.path.join(path_raw_data, f'1D/split_{i+1}.nc'),
            }
        else:
            paths = generate_simulation_paths(**simu_parameters, folder_path=path_raw_data)
            print(paths)

        simu = Simulation(data_folder_paths=[paths['path_1d'], paths['path_2d'], paths['path_3d']],
                          **simu_parameters)
        return simu
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the simulation paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
    
    return None


def plot_3d_visualization(i, simu_index, qn, vsurf, x, y, z):
    """
    Create 3D visualization for clouds and surface temperature.
    
    Parameters:
    - i: time index to display
    - simu_index: simulation index (for split number)
    - qn: cloud water data
    - vsurf: surface temperature data
    - x, y, z: coordinate arrays
    """
    
    # Prepare 3D variable for marching cubes (cloud isosurface)
    y2 = np.swapaxes(np.array(qn[i]), 0, 2)
    iso_val = 0.02
    verts, faces, _, _ = measure.marching_cubes(y2, level=iso_val)

    # Map grid coordinates
    verts[:, 0] = x[verts[:, 0].astype(int)]
    verts[:, 1] = y[verts[:, 1].astype(int)]
    verts[:, 2] = z[verts[:, 2].astype(int)]

    # Calculate surface temperature range
    min_tsurf = np.mean(vsurf[i]) - 1.5 * np.std(vsurf[i])
    max_tsurf = np.mean(vsurf[i]) + 1.5 * np.std(vsurf[i])

    # Create figure
    fig = plt.figure(figsize=(16, 9))
    ax_3d = fig.add_subplot(111, projection='3d')
    
    # Surface temperature background
    XX, YY = np.meshgrid(x, y)
    cset = ax_3d.contourf(XX, YY, vsurf[i], 25, zdir="z", offset=0, 
                         cmap='RdYlBu_r', alpha=0.9, zorder=0, 
                         vmin=min_tsurf, vmax=max_tsurf)

    # Cloud isosurface
    plot = ax_3d.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], 
        cmap='Blues', lw=1, alpha=0.5, zorder=100
    )

    # 3D plot settings
    ax_3d.set_box_aspect([1, 1, 0.2])
    ax_3d.set_xlabel("X [km]", labelpad=25)
    ax_3d.set_ylabel("Y [km]", labelpad=30)
    ax_3d.set_zlabel("Z [km]", labelpad=10, rotation=180)
    ax_3d.view_init(elev=12, azim=-60)
    ax_3d.set_zlim(0, 17)

    # Set background color
    color_background = (0.15294117647058825, 0.18823529411764706, 0.24313725490196078, 1.0)
    ax_3d.xaxis.set_pane_color(color_background)
    ax_3d.yaxis.set_pane_color(color_background)
    ax_3d.zaxis.set_pane_color(color_background)

    # Z-axis ticks
    tick_positions = [0, 7.5, 15]
    tick_labels = [str(int(pos)) for pos in tick_positions]
    ax_3d.set_zticks(tick_positions)
    ax_3d.set_zticklabels(tick_labels)

    # Temperature colorbar with 1 decimal place
    cb = fig.colorbar(cset, ax=ax_3d, shrink=0.4, aspect=100, pad=0.1, orientation='horizontal')
    cb.set_label('Temperature [K]')
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    # Title
    plt.title(f'Split {simu_index+1} - Time Index {i} - 3D Clouds & Surface Temperature', fontsize=16, pad=20)
    
    return fig


def plot_isentropic_visualization(i, simu_index, isentrop_1, isentrop_2, z):
    """
    Create visualization with two isentropic diagrams side by side.
    
    Parameters:
    - i: time index to display
    - simu_index: simulation index (for split number)
    - isentrop_1, isentrop_2: two different isentropic datasets to compare
    - z: coordinate array
    """
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Prepare coordinate grids for isentropic plots
    s = np.linspace(0, 1, 50)
    X2D, Y2D = np.meshgrid(s, z)

    # Isentropic diagram 1
    im1 = ax1.contourf(X2D, Y2D, isentrop_1[i], levels=20, cmap='RdBu_r', vmin=-50, vmax=50)
    ax1.set_title('Isentropic Diagram 1')
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Z [km]')
    ax1.set_ylim(0, 17)
    
    # Individual colorbar for first plot
    cb1 = fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.8)
    cb1.set_label('Mass Flux [kg.m/s]')
    
    # Isentropic diagram 2
    im2 = ax2.contourf(X2D, Y2D, isentrop_2[i], levels=20, cmap='RdBu_r', vmin=-50, vmax=50)
    ax2.set_title('Isentropic Diagram 2')
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Z [km]')
    ax2.set_ylim(0, 17)

    # Individual colorbar for second plot
    cb2 = fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.8)
    cb2.set_label('Mass Flux [kg.m/s]')

    # Overall title
    plt.suptitle(f'Split {simu_index+1} - Time Index {i} - Isentropic Analysis', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def save_figure_as_image(fig, filename, file_format='png'):
    """Save matplotlib figure as image file."""
    fig.savefig(f"{filename}.{file_format}", format=file_format, dpi=300, bbox_inches='tight')


def main():
    """Main function to run the visualization."""
    
    # Configuration
    simu_index = 15  # Which simulation split to load (0-based index)
    time_index = 200  # Which time step to display
    path_raw_data = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
    
    # Simulation parameters for the first split
    simu_parameters = {
        'velocity': '8',
        'temperature': '300', 
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': str(simu_index + 1)
    }
    
    print(f"Loading simulation split {simu_index + 1}...")
    
    # Load simulation
    simu = load_simulation(simu_parameters, i=simu_index, path_raw_data=path_raw_data)
    
    if not simu:
        print("Failed to load simulation!")
        return
    
    print(f"Simulation loaded: {simu.name}")
    
    # Load backup data if available
    backup_path = f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu'
    if os.path.exists(backup_path):
        print("Loading backup data...")
        simu.load(backup_folder_path=backup_path)
    
    # Extract data
    qn = simu.dataset_3d.QN.values
    vsurf = simu.dataset_3d.TABS[:, 0]  # Surface temperature
    x = simu.dataset_3d.x / 1000  # Convert to km
    y = simu.dataset_3d.y / 1000  # Convert to km  
    z = simu.dataset_3d.z / 1000  # Convert to km
    
    # Load two different isentropic datasets for comparison
    # You can modify these to load different variables
    isentrop_1 = simu.dataset_isentropic.RHO_W_sum.values  # First isentropic variable
    isentrop_2 = simu.dataset_isentropic.RHO_W_sum.values  # Second isentropic variable (same for now)
    
    # Check if time index is valid
    nt = qn.shape[0]
    if time_index >= nt:
        print(f"Time index {time_index} out of range. Dataset has {nt} time steps.")
        print(f"Using last available time step: {nt-1}")
        time_index = nt - 1
    
    print(f"Creating visualizations for time index {time_index}...")
    
    # Turn off interactive mode
    plt.ioff()
    
    # Create 3D visualization
    fig_3d = plot_3d_visualization(
        i=time_index,
        simu_index=simu_index,
        qn=qn,
        vsurf=vsurf, 
        x=x, y=y, z=z
    )
    
    # Create isentropic visualization
    fig_isentropic = plot_isentropic_visualization(
        i=time_index,
        simu_index=simu_index,
        isentrop_1=isentrop_1,
        isentrop_2=isentrop_2,
        z=z
    )
    
    # Show the plots
    plt.show()
    
    # Optionally save the figures
    save_option = input("Save figures? (y/n): ").lower().strip()
    if save_option == 'y':
        output_3d = f"split_{simu_index+1}_time_{time_index}_3d"
        output_isentropic = f"split_{simu_index+1}_time_{time_index}_isentropic"
        
        save_figure_as_image(fig_3d, output_3d, 'png')
        save_figure_as_image(fig_isentropic, output_isentropic, 'png')
        
        print(f"3D figure saved as {output_3d}.png")
        print(f"Isentropic figure saved as {output_isentropic}.png")


if __name__ == "__main__":
    main()
