import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage import measure

import numpy as np
from skimage import measure
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm
import multiprocessing as mp




# Ensure pySAMetrics is available in your Python environment
import pySAMetrics
from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths

from pySAMetrics.Simulation_2 import Simulation

data_dict = {}

for i in range(1, 49):
    data_dict[f'split_{i}'] = {
        'velocity': '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': str(i)
    }



def load_simulation(simu_parameters, i=1000, path_raw_data='/Volumes/LaCie/000_POSTDOC_2025/long_high_res'):
    """
    Load and run the simulation for the given parameters.

    Parameters:
    - simu_parameters (dict): Dictionary containing simulation parameters.

    Returns:
    - Simulation object or None if failed to load.
    """
    try:
        if i!=1000:
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker



def plot_visualisation_matplotlib(i, qn, mse, vsurf, x, y, z, isentrop, isentropic_levels=None):
    """
    Matplotlib version: 3D isosurface (left) + 2D isentropic contourf (right).
    """
    
    fig = plt.figure(figsize=(16, 9))

    gs = GridSpec(nrows=3, ncols=2, height_ratios=[0.1, 4, 0.1], width_ratios=[4, 1.5], hspace=0.3, figure=fig)

    # 3D plot spans all rows in column 0 (first column)
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.set_box_aspect(aspect=(1, 1, 0.3))


    # Access the panes (older method)
    ax3d.xaxis.set_pane_color((0.0, 0.12, 0.3, 1))
    ax3d.yaxis.set_pane_color((0.0, 0.12, 0.3, 1))
    #ax3d.zaxis.set_pane_color((1, 1, 1, 0))  # transparent
    ax3d.zaxis.set_pane_color((0.0, 0.12, 0.3, 1))



    # 2D plot spans rows 0 and 1 in column 1 (second column)
    ax2d = fig.add_subplot(gs[1:2, 1])

    # Colorbar axis at row 2 (bottom row), column 1
    cax = fig.add_subplot(gs[1, 1])

    # Surface temperature as colored ground plane
    XX, YY = np.meshgrid(x, y)
    #surf = ax3d.plot_surface(
    #    XX, YY, np.zeros_like(vsurf[i]),
    #    facecolors=plt.cm.RdYlBu_r((vsurf[i] - 296) / (300 - 296)),
    #    rstride=1, cstride=1, antialiased=False, shade=False
    #)

    # Prepare cloud water data
    y2 = np.swapaxes(np.array(qn[i]), 0, 2)
    iso_val = 0.02
    verts, faces, _, _ = measure.marching_cubes(y2, level=iso_val)

    # Map to actual coordinates
    verts[:, 0] = x[verts[:, 0].astype(int)]
    verts[:, 1] = y[verts[:, 1].astype(int)]
    verts[:, 2] = z[verts[:, 2].astype(int)]

    mesh = Poly3DCollection(verts[faces], alpha=0.4, facecolor='white', edgecolor='none')
    ax3d.add_collection3d(mesh)

    ax3d.set_xlim(x.min(), x.max())
    ax3d.set_ylim(y.min(), y.max())
    ax3d.set_zlim(0, 17)
    ax3d.set_xlabel('X [km]')
    ax3d.set_ylabel('Y [km]')
    ax3d.set_zlabel('Z [km]')
    ax3d.view_init(elev=20, azim=240)
    #ax3d.set_title("3D Isosurface + Surface Temp")

    ax3d.set_box_aspect(aspect=(1, 1, 0.3))

    # -------- 2D Isentropic Slice (right) --------
    #ax2d = fig.add_subplot(1, 2, 2)
    isentropic_slice = isentrop[i]  # shape: (nz, ny)

    if isentropic_levels is None:
        isentropic_levels = np.concatenate([
            np.linspace(-150, -50, 3),
            np.linspace(-50, -10, 8),
            np.linspace(-10, 10, 20),
            np.linspace(10, 50, 8),
            np.linspace(50, 150, 3)
        ])
        isentropic_levels = np.sort(np.unique(isentropic_levels))

    x_mse = np.linspace(pySAMetrics.FMSE_MIN, pySAMetrics.FMSE_MAX,50)/1000
    Y2d, Z2d = np.meshgrid(x_mse, z)


    cf = ax2d.contourf(Y2d, Z2d, isentropic_slice, levels=isentropic_levels, vmin=-40, vmax=40, cmap='RdBu_r', extend='both')
    cbar = plt.colorbar(cf, orientation='horizontal')
    cbar.set_label("Mass Flux [kg/m²/s]")
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    cax.set_axis_off()



    ax2d.set_ylim(0, 16)
    ax2d.set_xlabel("MSE [J/K/kg]")
    ax2d.set_ylabel("Z [km]")
    ax2d.set_title("Isentropic Mass Flux")
    ax2d.grid(True)
    plt.subplots_adjust(hspace=0.1)  # reduce vertical spacing between subplots


    plt.tight_layout()
    #plt.show()
    return fig


def save_figure_as_image(fig, filename, file_format='png', dpi=300, bbox_inches='tight'):
    fig.savefig(f"{filename}.{file_format}", format=file_format, dpi=dpi, bbox_inches=bbox_inches)


#list_files = [f'split_{i}' for i in range(4,8)]
list_files = [f'split_{4}']

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    for i_file,file in tqdm(enumerate(list_files, start=0)):
        print('Run', file)
        parameters = data_dict[file]
        
        simu = load_simulation(parameters, i=i_file)
        print(simu.name)
        
        print('hey hey!!!')
        print(simu.name)

        if simu:
            simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')


        print(simu.name)
        #simu.load("/Volumes/LaCie/000_POSTDOC_2025/control_short/saved_simu")

        
        qn = simu.dataset_3d.QN.values
        mse = simu.dataset_computed_3d.FMSE.values
        vsurf = simu.dataset_3d.TABS[:,0]
        isentrop = simu.dataset_isentropic.RHO_W_sum.values
        x = simu.dataset_3d.x/1000
        y = simu.dataset_3d.y/1000
        z = simu.dataset_3d.z/1000

        nt = qn.shape[0]

        #XX, YY = np.meshgrid(x,y)



        # Example usage: loop through time steps from 0 to 20
        #for i in tqdm(range(300,700)):  # from time step 0 to 20
        for i in tqdm(range(nt)):  # from time step 0 to nt
        #for i in tqdm(range(5)):  # from time step 0 to 20
        #for i in tqdm([77]):
            #fig_i = plot_figure_at_time(i, qn, mse, vsurf, x, y, z, XX, YY, isentrop)
            #fig_i = plot_visualisation_matplotlib(i=i, qn=qn, mse=mse, vsurf=vsurf, x=x, y=y, z=z, isentrop=isentrop)
            fig_i = plot_visualisation_matplotlib(i=i, qn=qn, mse=mse, vsurf=vsurf, x=x, y=y, z=z, isentrop=isentrop)

            frame_index = i_file * 1000 + i  # ensure unique + sortable across sims
            filename = os.path.join(
                '/Volumes/LaCie/000_POSTDOC_2025/new_img_simple_video',
                f'img_{str(frame_index).zfill(6)}')
            print(filename)
            save_figure_as_image(fig_i, filename, file_format='png')
            #save_figure_as_image(fig_i, filename, file_format='png')



