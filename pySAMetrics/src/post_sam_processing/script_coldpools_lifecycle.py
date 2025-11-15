from pySAMetrics.ColdPool import ColdPool, extract_cold_pools
from script_simu_high_Res_long import data_dict, load_simulation

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp



output_dir = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/'
list_files = [f'split_{i}' for i in range(4, 40)]

def cold_pools_to_dataframe(cold_pools, sim_name):
    return pd.DataFrame([{
        'simulation': sim_name,
        'label_id': cp.label_id,
        'start_time': cp.start_time,
        'end_time': cp.end_time,
        'duration': cp.duration,
        'start_size': cp.start_size,
        'end_size': cp.end_size,
        'max_size': cp.max_size,
        'mean_qv': cp.mean_qv,
        'mean_domain_qv': cp.mean_domain_qv,
        'anomaly_qv': cp.anomaly_qv,
        'max_anomaly_qv': cp.max_anomaly_qv
    } for cp in cold_pools])

def extract_cold_pools_with_timeseries(label_array, qv_array, global_offset):
    """
    Extends extract_cold_pools to also return time-resolved rows with global timestep.
    """
    from collections import defaultdict

    nt = label_array.shape[0]
    label_info = defaultdict(lambda: {
        'timesteps': [], 'sizes': [], 'qv_values': [],
        'domain_qv': [], 'qv_anomalies': [],
        'evolution': []
    })

    for t in range(nt):
        frame = label_array[t]
        qv = qv_array[t]
        domain_mean_qv = np.mean(qv)

        for label in np.unique(frame):
            if label <= 0:
                continue
            mask = frame == label
            qv_vals = qv[mask]
            mean_qv_in_cp = np.mean(qv_vals)
            anomaly = mean_qv_in_cp - domain_mean_qv

            label_info[label]['timesteps'].append(t)
            label_info[label]['sizes'].append(np.sum(mask))
            label_info[label]['qv_values'].extend(qv_vals.tolist())
            label_info[label]['domain_qv'].append(domain_mean_qv)
            label_info[label]['qv_anomalies'].append(anomaly)

            label_info[label]['evolution'].append({
                'time': t,
                'global_time': t + global_offset,
                'size': np.sum(mask),
                'qv_mean_cp': mean_qv_in_cp,
                'qv_mean_domain': domain_mean_qv,
                'anomaly_qv': anomaly
            })

    cold_pools = []
    timeseries_rows = []

    for label, info in label_info.items():
        cp = ColdPool(
            label_id=label,
            timesteps=info['timesteps'],
            sizes=info['sizes'],
            qv_values=info['qv_values'],
            domain_qv_means=info['domain_qv'],
            qv_anomalies=info['qv_anomalies']
        )
        cold_pools.append(cp)

        for evo in info['evolution']:
            evo['label_id'] = label
            timeseries_rows.append(evo)

    return cold_pools, timeseries_rows, nt  # include number of time steps for offset tracking

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    all_summaries = []
    all_timeseries = []
    global_offset = 0

    for i_file, file in tqdm(enumerate(list_files, start=4)):
        print('Run', file)
        parameters = data_dict[file]
        simu = load_simulation(parameters, i=i_file)
        
        print(simu.name)
        if simu:
            simu.load(backup_folder_path=f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu')

        print(simu.name)

        label_array = simu.dataset_computed_2d.CP_LABELS.values  # (nt, nx, ny)
        qv_array = simu.dataset_3d.QV[:, 0].values  # (nt, nx, ny)

        cold_pools, timeseries_rows, nt = extract_cold_pools_with_timeseries(
            label_array, qv_array, global_offset
        )

        # Store with simulation name
        summary_df = cold_pools_to_dataframe(cold_pools, simu.name)
        timeseries_df = pd.DataFrame(timeseries_rows)
        timeseries_df['simulation'] = simu.name

        all_summaries.append(summary_df)
        all_timeseries.append(timeseries_df)

        global_offset += nt  # update global time offset

    # Save everything
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "cold_pools_summary_ALL.csv")
    timeseries_path = os.path.join(output_dir, "cold_pools_timeseries_ALL.csv")

    pd.concat(all_summaries, ignore_index=True).to_csv(summary_path, index=False)
    pd.concat(all_timeseries, ignore_index=True).to_csv(timeseries_path, index=False)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved time series to: {timeseries_path}")
