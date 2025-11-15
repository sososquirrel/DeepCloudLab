#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import multiprocessing as mp

from script_simu_high_Res_long import data_dict, load_simulation

############################################
# Parameters
############################################
prec_threshold = 0.5      # mm/hr threshold to define rainy pixels
min_points = 20           # to trust statistics

list_files = [f'split_{i}' for i in range(4, 40)]
output_dir = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/'

# Storage
Iorg_all = []
beta_all = []
shallow_frac = []
deep_frac = []

############################################
# Helper: compute spacing stats
############################################
def compute_Iorg_beta(mask):
    # mask shape (nx, ny)
    pts = np.column_stack(np.where(mask))

    if pts.shape[0] < min_points:
        return np.nan, np.nan

    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    nn = dist[:, 1]

    # compare NN distribution to Poisson
    # (Bony & Tompkins use CDF ratio)
    R = np.sort(nn)
    CDF = np.arange(1, len(nn)+1) / len(nn)

    # normalized coordinate
    lam = len(nn) / mask.size
    R_exp = 1 - np.exp(-lam * np.pi * R**2)

    Iorg = np.mean(CDF > R_exp)
    beta = np.mean(R / np.mean(R))

    return Iorg, beta

############################################
# Helper: shallow/deep fractions
############################################
def classify_shallow_deep(PW):
    p40 = np.percentile(PW, 40)
    p60 = np.percentile(PW, 60)

    shallow = PW < p40
    deep = PW > p60

    return shallow.mean(), deep.mean()

############################################
# Main
############################################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for i_file, file in tqdm(enumerate(list_files, start=4)):
        parameters = data_dict[file]
        simu = load_simulation(parameters, i=i_file)

        Prec = simu.dataset_2d.Prec.values        # (nt,nx,ny)
        PW   = simu.dataset_2d.PW.values

        nt = Prec.shape[0]

        for t in range(nt):
            rain_mask = Prec[t] > prec_threshold

            Iorg, beta = compute_Iorg_beta(rain_mask)
            Iorg_all.append(Iorg)
            beta_all.append(beta)

            s, d = classify_shallow_deep(PW[t])
            shallow_frac.append(s)
            deep_frac.append(d)

    # Save
    np.save(os.path.join(output_dir, 'Iorg.npy'), np.array(Iorg_all))
    np.save(os.path.join(output_dir, 'beta.npy'), np.array(beta_all))
    np.save(os.path.join(output_dir, 'shallow_frac.npy'), np.array(shallow_frac))
    np.save(os.path.join(output_dir, 'deep_frac.npy'), np.array(deep_frac))
