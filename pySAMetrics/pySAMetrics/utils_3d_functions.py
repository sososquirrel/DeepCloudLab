import numpy as np
import xarray as xr
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm

###DATA


def get_3d_image(xr_var_2d, xr_var_3d,i_time,x, y, z, outname):
    if isinstance(xr_var_3d, np.ndarray):
        xr_var_3d_i = xr_var_3d[i_time]
        xr_var_2d_i = xr_var_2d.isel(time=i_time)

    else:
        xr_var_2d_i = xr_var_2d.isel(time=i_time)
        xr_var_3d_i = xr_var_3d.isel(time=i_time)


    ##GET VAR

    
    #q = simulation.dataset_3d.QN[:, :35, :, :].values
    xr_var_3d_i = np.array(xr_var_3d_i)
    
    y2 = np.swapaxes(xr_var_3d_i, 0, 2)

    #y2 = y2[0]
    #t_surf = simulation.dataset_2d.TSFCi.values
    v_surf = xr_var_2d_i
    min_tsurf = np.mean(v_surf)-1.5*np.std(v_surf)
    max_tsurf = np.mean(v_surf)+1.5*np.std(v_surf)

    #y2_smoothed = gaussian_filter(y2, sigma=0.5)  # Adjust sigma for smoothness


    ##CALCULATE SURF

    #iso_val =  0.03
    iso_val=1
    verts, faces, normals, values = measure.marching_cubes(y2, iso_val)

    ##FOR PLOT
    n_bins=10
    cmap=plt.get_cmap('Blues')
    cmap2=plt.get_cmap('jet')

    colors_cloud=[cmap(0.), cmap(0.1),  cmap(0.2)]
    cm_cloud = LinearSegmentedColormap.from_list('cloud', colors_cloud, N=n_bins)


    ###PLOT
    fig1 = plt.figure(figsize=(20, 9))
    ax = fig1.add_subplot(111, projection="3d")

    XX, YY = np.meshgrid(x/1000, y/1000)

    cset = ax.contourf(XX, YY, v_surf.values, 25, zdir="z", offset=0, cmap='RdBu_r', alpha=0.9, zorder=0, vmin=min_tsurf, vmax=max_tsurf)


    plot = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], 
        cmap='Blues', lw=1, alpha=0.5, zorder=100
    )

    ax.set_box_aspect([1,1,0.2])


    ax.set_xlabel("y [km]", labelpad=25)
    ax.set_ylabel("x [km]",  labelpad=30)
    ax.set_zlabel("z [km]",  labelpad=10, rotation=180)



    #ax.azim =-60
    #ax.dist = 10
    #ax.elev = 12

    ax.view_init(elev=12, azim=-60)


    color_background=(0.15294117647058825, 0.18823529411764706, 0.24313725490196078, 1.0)
    ax.w_xaxis.set_pane_color(color_background)
    ax.w_yaxis.set_pane_color(color_background)
    ax.w_zaxis.set_pane_color(color_background)

    ax.set_zlim(0,53)


    tick_positions = [0, 10, 20, 30, 40, 50]
    zarr = z
    tick_labels = [np.round(zarr[int(pos)]/1000,0) for pos in tick_positions]
    ax.set_zticks(tick_positions)
    ax.set_zticklabels(tick_labels)

    cb=fig1.colorbar(cset, shrink=0.4, aspect=100, pad=-0.3, orientation='horizontal')
    cb.set_label(label='Temperature [K]')

    # Remove the spines to make the plot's box invisible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    
    #plt.tight_layout()
    
    plt.savefig(f'{outname}.jpeg', format='jpeg', dpi=300, pad_inches=0, transparent=True)
    img = Image.open(f'{outname}.jpeg')
    crop_box = (1800, 1000, 4600, 2300)
    cropped_img = img.crop(crop_box)
    print(f'{outname}.jpeg')
    cropped_img.save(f'{outname}.jpeg')
    plt.close(fig1)

def get_sequence_images(i_start, i_stop, xr_var_2d, xr_var_3d,x, y, z, outname):
    for i in tqdm(range(i_start, i_stop)):
        get_3d_image(xr_var_2d=xr_var_2d, 
                     xr_var_3d=xr_var_3d,
                     i_time=i,
                     x=x,
                     y=y,
                     z=z,
                     outname=f'{outname}_{str(i).zfill(4)}')
        






