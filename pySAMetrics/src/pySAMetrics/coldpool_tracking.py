import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
import xarray as xr
import copy
import gc

# ------------------------
# Public API function
# ------------------------
def get_coldpool_tracking_images(simulation, variable_images, low_threshold, high_threshold,
                                 n_jobs=4, use_loky=True):
    """
    Detect cold-pools per time slice and track labels in time.

    Uses joblib Parallel with loky backend (process-based) by default for robust
    multiprocessing on macOS. Falls back to simple serial loop if n_jobs == 1.

    Parameters
    ----------
    simulation : object
        Must expose .nt (number of time steps).
    variable_images : np.ndarray
        3D array (time, y, x) of the diagnostic used for detection.
    low_threshold, high_threshold : float
        thresholds passed to generate_cluster_labels
    n_jobs : int
        Number of worker processes (set to 1 to run serial).
    use_loky : bool
        If True, use loky backend (recommended). If False, joblib will pick a backend.
    """
    # Build list of 2D images (one per time step)
    nt = int(simulation.nt)
    image_slices = [variable_images[t, :, :] for t in range(nt)]

    # Partial function with thresholds baked in
    process_func = partial(process_variable_images,
                           low_threshold=low_threshold, high_threshold=high_threshold)

    # Run in parallel (joblib + loky recommended). If n_jobs==1 do serial to avoid overhead.
    if n_jobs == 1:
        results = [process_func(img) for img in tqdm(image_slices, desc="processing (serial)")]
    else:
        backend = "loky" if use_loky else None
        # Wrap generator with tqdm so we show progress
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(process_func)(img) for img in tqdm(image_slices, desc="processing", total=nt)
        )

    # Convert results to numpy array: shape (nt, 3, y, x)
    results = np.array(results, dtype=object)  # object first, then unpack below

    # Unpack sequences
    core_sequence = np.stack(results[:, 0])
    envelop_sequence = np.stack(results[:, 1])
    labeled_total = np.stack(results[:, 2])

    # binary flags for core/envelope (any non-zero -> 1)
    core_binary = (core_sequence != 0).astype(np.uint8)
    envelop_binary = (envelop_sequence != 0).astype(np.uint8)

    # attach to simulation
    simulation.label_total = labeled_total

    # Build list for tracking
    labeled_image_seq = [l for l in labeled_total]

    SIMILARITY_THRESHOLD = 0.55
    tracking_cp = track_clusters_over_time(list_labeled_image=labeled_image_seq,
                                           similarity_threshold=SIMILARITY_THRESHOLD)

    # Make CP binary mask
    cp_binary = (labeled_total != 0).astype(np.uint8)

    # Assign / create dataset_computed_2d as before
    ylen = core_binary.shape[1]
    xlen = core_binary.shape[2]
    times = range(core_binary.shape[0])

    ds = getattr(simulation, "dataset_computed_2d", None)
    if ds is None:
        ds = xr.Dataset(coords={"time": times, "y": range(ylen), "x": range(xlen)})

    ds = ds.assign(
        CORE_BINARY=(("time", "y", "x"), core_binary, {"long_name": "Core Cold Pool location", "units": "1"})
    )
    ds = ds.assign(
        ENVELOP_BINARY=(("time", "y", "x"), envelop_binary, {"long_name": "Envelop Cold Pool location", "units": "1"})
    )
    ds = ds.assign(
        CP_BINARY=(("time", "y", "x"), cp_binary, {"long_name": "Cold Pool location", "units": "1"})
    )
    ds = ds.assign(
        CP_LABELS=(("time", "y", "x"), tracking_cp, {"long_name": "Cold Pool label", "units": "1"})
    )

    simulation.dataset_computed_2d = ds

    # cleanup large temporaries
    del results, image_slices
    gc.collect()

# ------------------------
# Helper functions (unchanged, only minor style tweaks)
# ------------------------
def process_variable_images(image_2d, low_threshold, high_threshold):
    labeled_core_image, labeled_envelop_image, labeled_total = generate_cluster_labels(
        image_2d, low_threshold, high_threshold
    )
    return labeled_core_image, labeled_envelop_image, labeled_total


def periodic_distance(A, B):
    x1, y1 = A
    x2, y2 = B
    dx = min(abs(x2 - x1), abs(x1 + 128 - x2))
    dy = min(abs(y2 - y1), abs(y1 + 128 - y2))
    return dx**2 + dy**2


def apply_kmeans_to_variable(i_image, sigma=3, n_clusters=2):
    i_image = gaussian_filter(i_image.astype(float), sigma=sigma)
    variable_flat = i_image.flatten().reshape(-1, 1)

    kmeans_level_1 = KMeans(n_clusters=n_clusters, random_state=0, n_init=1)
    kmeans_level_1.fit(variable_flat)
    labels_level_1 = kmeans_level_1.labels_.reshape(i_image.shape)

    # Ensure label=1 corresponds to higher values
    if np.min(i_image[labels_level_1 == 1]) > np.min(i_image[labels_level_1 == 0]):
        labels_level_1 = 1 - labels_level_1

    variable_values_labeled_1 = i_image[labels_level_1 == 1].reshape(-1, 1)
    if variable_values_labeled_1.size == 0:
        labels_level_2 = np.copy(labels_level_1)
    else:
        kmeans_level_2 = KMeans(n_clusters=n_clusters, random_state=0, n_init=1)
        new_labels = kmeans_level_2.fit_predict(variable_values_labeled_1)
        labels_level_2 = np.copy(labels_level_1)
        labels_level_2[labels_level_1 == 1] = new_labels

    return labels_level_1, labels_level_2


def create_binary_image(i_image, threshold, sigma=2):
    i_image = gaussian_filter(i_image.astype(float), sigma=sigma)
    return (i_image < threshold).astype(float)


def generate_cluster_labels(i_image, low_threshold, high_threshold):
    low_threshold_binary_image = create_binary_image(i_image, threshold=low_threshold)
    high_threshold_binary_image = create_binary_image(i_image, threshold=high_threshold)

    ensemble_idx = np.array(np.where(low_threshold_binary_image)).T
    if ensemble_idx.size == 0:
        return np.zeros_like(i_image), np.zeros_like(i_image), np.zeros_like(i_image)

    # pairwise distances (periodic)
    distance_matrix_ensemble = pairwise_distances(ensemble_idx, metric=periodic_distance)
    if distance_matrix_ensemble.shape[0] < 2:
        labeled_core_image = np.zeros_like(i_image)
        labeled_envelop_image = np.zeros_like(i_image)
        labeled_total = np.zeros_like(i_image)
        labeled_ensemble_image = np.zeros_like(i_image)
        labeled_ensemble_image[tuple(ensemble_idx[0])] = 1
        return labeled_core_image, labeled_envelop_image, labeled_total

    clustering_ensemble = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.1, linkage="single", metric="precomputed"
    ).fit(distance_matrix_ensemble)

    labeled_ensemble_image = np.zeros_like(i_image)
    labeled_core_image = np.zeros_like(i_image)
    labeled_ensemble_image[np.where(low_threshold_binary_image)] = clustering_ensemble.labels_ + 1

    start_label_core = 1
    labeled_envelop_image = np.zeros_like(i_image)

    for label_ensemble in np.unique(labeled_ensemble_image):
        if label_ensemble == 0:
            continue
        idx_core_ensemble = np.where((labeled_ensemble_image == label_ensemble) & (high_threshold_binary_image == 1))
        idx_core_ensemble_reshape = np.array(idx_core_ensemble).T
        idx_ensemble_label = np.where(labeled_ensemble_image == label_ensemble)

        if idx_core_ensemble_reshape.size <= 2:
            labeled_ensemble_image[idx_ensemble_label] = 0
            labeled_core_image[idx_ensemble_label] = 0
            labeled_envelop_image[idx_ensemble_label] = 0
        else:
            distance_matrix_core = pairwise_distances(idx_core_ensemble_reshape, metric=periodic_distance)
            clustering_core_ensemble = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.1, linkage="single", metric="precomputed"
            ).fit(distance_matrix_core)

            labeled_core_image[idx_core_ensemble] = clustering_core_ensemble.labels_ + start_label_core
            start_label_core += np.max(clustering_core_ensemble.labels_) + 1

            X = np.array(idx_core_ensemble).T
            y = labeled_core_image[idx_core_ensemble]
            knn = KNeighborsClassifier(n_neighbors=1, metric=periodic_distance)
            knn.fit(X, y)

            X_ens = np.array(idx_ensemble_label).T
            new_predictions = knn.predict(X_ens)
            labeled_envelop_image[idx_ensemble_label] = new_predictions

    labeled_envelop_image -= labeled_core_image
    labeled_total = labeled_core_image + labeled_envelop_image

    return labeled_core_image, labeled_envelop_image, labeled_total


def measure_intersection(image1, label1, image2, label2):
    intersection_indices = np.where((image1 == label1) & (image2 == label2))
    intersection_count = len(intersection_indices[0])
    total_label1_count = len(np.where(image1 == label1)[0])
    return intersection_count / total_label1_count if total_label1_count > 0 else 0.0


def track_clusters_over_time(list_labeled_image, similarity_threshold):
    image_1 = list_labeled_image[0]
    list_new_seq = [image_1]

    for i_image in range(1, len(list_labeled_image)):
        max_label_step_t = np.max(image_1) + 10000
        image_2 = list_labeled_image[i_image] + max_label_step_t
        image_2[image_2 == max_label_step_t] = 0

        list_labels_1 = list(np.unique(image_1))
        if 0 in list_labels_1:
            list_labels_1.remove(0)
        list_labels_2 = list(np.unique(image_2))
        if 0 in list_labels_2:
            list_labels_2.remove(0)

        if (len(list_labels_1) == 0) or (len(list_labels_2) == 0):
            output_image_step_tt = image_2
        else:
            similarity_matrix = np.zeros((len(list_labels_1), len(list_labels_2)))
            output_image_step_tt = copy.deepcopy(image_2)

            for i, label1 in enumerate(list_labels_1):
                for j, label2 in enumerate(list_labels_2):
                    similarity_matrix[i, j] = measure_intersection(
                        image1=image_1, label1=label1, image2=image_2, label2=label2
                    )

            for j, label2 in enumerate(list_labels_2):
                new_label_j = list_labels_1[np.argmax(similarity_matrix, axis=0)[j]]
                if np.max(similarity_matrix, axis=0)[j] > similarity_threshold:
                    output_image_step_tt[image_2 == label2] = new_label_j

        list_new_seq.append(output_image_step_tt)
        image_1 = output_image_step_tt

    input_array = np.array(list_new_seq)
    relabeled_labeled_image_seq = np.zeros_like(input_array)
    for i, label in enumerate(np.unique(input_array)):
        relabeled_labeled_image_seq[input_array == label] = i

    return relabeled_labeled_image_seq
