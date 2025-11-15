#n_t = simulation.dataset_3d.time.values.shape[0]
from tqdm import tqdm
import numpy as np
import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import pySAMetrics
from pySAMetrics.Simulation import Simulation
import matplotlib.gridspec as gridspec
from pySAMetrics.utils import generate_simulation_paths






for simu in list_simu:
    path_dir_img = '/burg/glab_new/users/sga2133/image_isentropic/isen_{simu.name}'
    os.makedirs(path_dir_img, exist_ok=True)

    n_t=300
    for i_time in tqdm(range(n_t)):
        # Create a figure
        fig = plt.figure(figsize=(20, 20), constrained_layout=True)

        gs = gridspec.GridSpec(5, 2, height_ratios=[3, 5, 0.1, 5, 0.1], width_ratios=[0.5, 0.5])

        # Add subplots
        # Large panel on the top (spans both columns)
        ax1 = fig.add_subplot(gs[0, :])

        # Two small panels on the left below the large panel
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[3, 0])

        # One larger panel on the right
        ax4 = fig.add_subplot(gs[1:-1, 1])

        # Colorbar space below the small panels and the right panel
        cax1 = fig.add_subplot(gs[2, 0])
        cax2 = fig.add_subplot(gs[4, 0])
        cax3 = fig.add_subplot(gs[4, 1])



        PW = simu.dataset_computed_3d.RHO_W.values[i_time]
        X, Y = simu.dataset_3d.x.values/1000, simu.dataset_3d.y.values/1000
        XX,YY = np.meshgrid(X,Y)
        im2 = ax2.pcolormesh(XX,YY,PW[40], cmap='Spectral_r')
        ax2.set_xlabel('x [km]')
        ax2.set_ylabel('y [km]')
        cbar2 = plt.colorbar(im2, cax=cax1, orientation='horizontal', label="Vertical Mass Flux at 13km [kg.m$^{-2}$.s$^{-1}$]")

        ax2.grid(True)


        BUOYANCY= simu.dataset_computed_3d.BUOYANCY[i_time]
        CP = simu.dataset_computed_2d.CP_LABELS.values[i_time]
        CP[CP==0]=np.nan
        X, Y = simu.dataset_3d.x.values/1000, simu.dataset_3d.y.values/1000
        XX,YY = np.meshgrid(X,Y)
        im3 = ax3.pcolormesh(XX,YY,BUOYANCY[0], vmin=-0.1, vmax=0.05, cmap='coolwarm')
        ax3.pcolormesh(XX,YY,CP%8, alpha=0.8, vmin=0, vmax=7, cmap='Set2')

        ax3.set_xlabel('x [km]')
        ax3.set_ylabel('y [km]')
        cbar3 = plt.colorbar(im3, cax=cax2, orientation='horizontal', label='Buoyancy [kJ/kg]')

        ax3.grid(True)

        fmse = np.linspace(320, 350, 50)
        z = simu.dataset_3d.z.values/1000
        FF, ZZ = np.meshgrid(fmse, z)
        fmse_z_matrix = simu.dataset_isentropic.RHO_W.values[i_time]
        im4 = ax4.pcolormesh(FF, ZZ, fmse_z_matrix, vmin=-2, vmax=15, cmap='rainbow')
        ax4.set_xlim(320, 350)
        ax4.set_ylim(0,16)
        ax4.set_xlabel('FMSE')
        ax4.set_ylabel('z [km]')
        cbar4 = plt.colorbar(im4, cax=cax3, orientation='horizontal', label=r"$\rho$ W [kg.m$^{-2}$.s$^{-1}$]")
        ax4.grid(True)

        fmse_in_time = np.var(simu.dataset_2d.PW.values, axis=(-1,-2))

        ax1.plot(simu.dataset_3d.time.values, fmse_in_time)
        ax1.scatter(simu.dataset_3d.time.values[i_time],fmse_in_time[i_time], marker='+', c='r')
        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel(r'$\mathrm{std}^2$(PW) [mm$^2$]')
        ax1.grid(True)



        path_fig = os.path.join(path_dir_img, f'{simu.name}_multi_plot_{str(i_time).zfill(4)}')
        plt.savefig(path_fig)

