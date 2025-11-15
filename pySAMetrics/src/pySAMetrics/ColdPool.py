import numpy as np
from collections import defaultdict

class ColdPool:
    def __init__(self, label_id, timesteps, sizes, qv_values, domain_qv_means, qv_anomalies):
        self.label_id = label_id
        self.start_time = timesteps[0]
        self.end_time = timesteps[-1]
        self.duration = self.end_time - self.start_time + 1
        self.start_size = sizes[0]
        self.end_size = sizes[-1]
        self.max_size = max(sizes)

        self.mean_qv = np.mean(qv_values)
        self.mean_domain_qv = np.mean(domain_qv_means)
        self.anomaly_qv = self.mean_qv - self.mean_domain_qv
        self.max_anomaly_qv = np.max(np.abs(qv_anomalies))  # max absolute anomaly

        self.cluster = {
            'timesteps': timesteps,
            'sizes': sizes,
            'qv_anomalies': qv_anomalies
        }

    def __repr__(self):
        return (f"ColdPool(label={self.label_id}, duration={self.duration}, "
                f"max_size={self.max_size}, anomaly_qv={self.anomaly_qv:.3e}, "
                f"max_anomaly_qv={self.max_anomaly_qv:.3e})")


def extract_cold_pools(label_array, qv_array):
    """
    label_array: (nt, nx, ny) - object labels
    qv_array: (nt, nx, ny) - specific humidity (same shape)
    """
    assert label_array.shape == qv_array.shape, "Shape mismatch"

    nt = label_array.shape[0]
    label_info = defaultdict(lambda: {
        'timesteps': [], 'sizes': [], 'qv_values': [],
        'domain_qv': [], 'qv_anomalies': []
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

            label_info[label]['timesteps'].append(t)
            label_info[label]['sizes'].append(np.sum(mask))
            label_info[label]['qv_values'].extend(qv_vals.tolist())
            label_info[label]['domain_qv'].append(domain_mean_qv)
            label_info[label]['qv_anomalies'].append(mean_qv_in_cp - domain_mean_qv)

    cold_pools = []
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

    return cold_pools
