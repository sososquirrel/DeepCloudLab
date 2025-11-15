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



# Ensure pySAMetrics is available in your Python environment
import pySAMetrics
from pySAMetrics.Simulation_2 import Simulation
from pySAMetrics.utils import generate_simulation_paths

# Initialize the main dictionary

data_dict = {
    'squall_line': {'velocity': '7.5', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'},
     'control_short': {'velocity': '0', 'temperature': '300', 'bowen_ratio': '1', 'microphysic': '1'}
    }


keys = ['control_short'] #['squall_line', 'control']

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



def plot_visualisation(i, qn, mse, vsurf, x, y, z, isentrop):
    # Sum cloud water over the vertical
    sum_qn = np.sum(qn[i], axis=2)
    
    # Moist static energy max selection
    subset = mse[i, :50]
    max_mse = subset[
        np.arange(subset.shape[0])[:, None], 
        np.arange(subset.shape[1]), 
        np.abs(subset).argmax(axis=2)
    ]
    
    #mse_plot = np.copy(max_mse)
    #mse_plot[mse_plot > 350000] = 350000
    
    # Prepare 3D variable for marching cubes
    y2 = np.swapaxes(np.array(qn[i]), 0, 2)
    iso_val = 0.02
    verts, faces, _, _ = measure.marching_cubes(y2, level=iso_val)

    # Map grid coordinates
    verts[:, 0] = x[verts[:, 0].astype(int)]
    verts[:, 1] = y[verts[:, 1].astype(int)]
    verts[:, 2] = z[verts[:, 2].astype(int)]

    # Surface background
    XX, YY = np.meshgrid(x, y)
    surf = go.Surface(
        z=np.zeros_like(vsurf[i]),
        x=XX,
        y=YY,
        surfacecolor=vsurf[i],
        cmin=296,
        cmax=300,
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(
            title=dict(text="Temperature [K]", side="top"),
            orientation="h", len=0.2, x=0.18, y=0.8
        ),
    )

    # Triangle mesh for isosurface
    trisurf = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='white',
        opacity=0.4,
        flatshading=True,
        showscale=False
    )

    # Vertical plane at x = constant
    x_plane = 64
    Y_plane, Z_plane = np.meshgrid(y, z)
    plane = go.Surface(
        x=np.full_like(Y_plane, x_plane),
        y=Y_plane,
        z=Z_plane,
        opacity=0.2,
        colorscale=[[0, 'blue'], [1, 'blue']],
        showscale=False,
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[0.42, 0.3, 0.25],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "contour"}, {"type": "xy", "rowspan": 2}],
            [None, {"type": "contour"}, None]
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.2,
        subplot_titles=("", "", "", "", "")
    )

    # Cloud water 2D slice
    Y2D, Z2D = np.meshgrid(y, z)
    fig.add_trace(go.Heatmap(
        z=sum_qn, y=Z2D[:, 0], x=Y2D[0],
        colorscale='RdBu_r',
        zmin=-1,
        zmax=15,
        showscale=True,
        colorbar=dict(
            title=dict(text="Cloud Fraction [g/kg]", side="top"),
            orientation="h",
            x=0.58,
            xanchor="center",
            y=0.42,
            len=0.2
        )
    ), row=1, col=2)

    # MSE 2D anomaly
    fig.add_trace(go.Heatmap(
        z=max_mse - np.mean(max_mse, axis=1, keepdims=True), 
        y=Z2D[:, 0], x=Y2D[0],
        colorscale='RdBu_r',
        zmax=350000,
        showscale=True,
        colorbar=dict(
            title=dict(text="Moist Static Energy Anomalie [J/kg]", side="top"),
            orientation="h",
            x=0.58,
            xanchor="center",
            y=-0.2,
            len=0.2
        )
    ), row=2, col=2)

    # 3D plots
    fig.add_trace(surf, row=1, col=1)
    fig.add_trace(trisurf, row=1, col=1)
    fig.add_trace(plane, row=1, col=1)

    # Isentropic vertical slice
    fig.add_trace(go.Heatmap(
        z=isentrop[i], y=Z2D[:, 0],
        colorscale='RdBu_r',
        zmin=-50,
        zmax=50,
        colorbar=dict(
            title=dict(text="Mass Flux [kg.m/s]", side="top"),
            orientation="h",
            len=0.2,
            x=0.88,
            y=-0.2
        )
    ), row=1, col=3)

    fig.update_yaxes(range=[0, 17], row=1, col=3)
    fig.update_yaxes(range=[0, 17], row=1, col=2)
    fig.update_yaxes(range=[0, 17], row=2, col=2)
    

    # Layout
    fig.update_layout(
        height=700,
        width=1700,
        title_text=f"Isentropic Analysis at Time {i}",
        showlegend=False,
        scene=dict(
            xaxis=dict(title='X [km]', backgroundcolor='rgba(39, 48, 62, 1)', gridcolor='white'),
            yaxis=dict(title='Y [km]', backgroundcolor='rgba(39, 48, 62, 1)', gridcolor='white'),
            zaxis=dict(title='Z [m]', range=[0, 17], tickvals=[0, 7.5, 15], backgroundcolor='rgba(39, 48, 62, 1)', gridcolor='white'),
            aspectratio=dict(x=1, y=1, z=0.2),
            camera=dict(eye=dict(x=1.4, y=-0.4, z=0.9))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    #fig.show()
    return fig


def save_figure_as_image(fig, filename, file_format='png'):
    fig.write_image(f"{filename}.{file_format}", format=file_format)



if __name__ == "__main__":

    data_dict = {}

    for i in range(1, 49):
        data_dict[f'split_{i}'] = {
            'velocity': '8',
            'temperature': '300',
            'bowen_ratio': '1',
            'microphysic': '1',
            'split': str(i)
        }

    parameters = data_dict[f'split_{10}']

    simu = load_simulation(parameters, i=10)
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

    #XX, YY = np.meshgrid(x,y)



    # Example usage: loop through time steps from 0 to 20
    #for i in tqdm(range(300,700)):  # from time step 0 to 20
    for i in tqdm(range(10,20)):  # from time step 0 to 20
        #fig_i = plot_figure_at_time(i, qn, mse, vsurf, x, y, z, XX, YY, isentrop)
        fig_i = plot_visualisation(i=i, qn=qn, mse=mse, vsurf=vsurf, x=x, y=y, z=z, isentrop=isentrop)
        filename = os.path.join('/Users/sophieabramian/Documents/DeepCloudLab/pySAMetrics/notebooks/script_video_10', f'img_{str(i).zfill(3)}')
        save_figure_as_image(fig_i, filename, file_format='png')



