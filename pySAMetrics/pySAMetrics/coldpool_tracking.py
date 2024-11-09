import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
import copy


def periodic_distance(A, B):
    """Compute the periodic squared distance between two points A and B."""
    x1, y1 = A
    x2, y2 = B
    
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    dx = min(max_x - min_x, min_x + 128 - max_x)
    dy = min(max_y - min_y, min_y + 128 - max_y)

    return dx**2 + dy**2


def apply_kmeans_to_variable(i_image, sigma=3, n_clusters=2):
    """
    Apply two levels of K-means clustering on the image after smoothing it.
    
    Parameters:
        i_image (np.ndarray): Input image representing any chosen variable.
        sigma (float): Standard deviation for Gaussian filter smoothing.
        n_clusters (int): Number of clusters for k-means.
    
    Returns:
        labels_level_1 (np.ndarray): Clustering labels after first level of k-means.
        labels_level_2 (np.ndarray): Clustering labels after second level of k-means.
    """
    i_image = gaussian_filter(i_image.astype(float), sigma=sigma)
    variable_flat = i_image.flatten().reshape(-1, 1)

    # First level of K-means clustering
    kmeans_level_1 = KMeans(n_clusters=n_clusters, random_state=0, n_init=1)
    kmeans_level_1.fit(variable_flat)
    labels_level_1 = kmeans_level_1.labels_.reshape(i_image.shape)
    
    # Ensure that label=1 corresponds to higher values of the variable
    if np.min(i_image[labels_level_1 == 1]) > np.min(i_image[labels_level_1 == 0]):
        labels_level_1 = 1 - labels_level_1  # Flip labels if necessary

    variable_values_labeled_1 = i_image[labels_level_1 == 1].reshape(-1, 1)

    if variable_values_labeled_1.size == 0:
        print("No points labeled as 1. Skipping second level of clustering.")
        labels_level_2 = np.copy(labels_level_1)
    else:
        # Second level of K-means clustering within points labeled as 1
        kmeans_level_2 = KMeans(n_clusters=n_clusters, random_state=0, n_init=1)
        new_labels = kmeans_level_2.fit_predict(variable_values_labeled_1)
        labels_level_2 = np.copy(labels_level_1)
        labels_level_2[labels_level_1 == 1] = new_labels

    return labels_level_1, labels_level_2


def create_binary_image(i_image, threshold, sigma=2):
    """
    Create a binary image by applying a Gaussian filter and thresholding.

    Parameters:
        i_image (np.ndarray): Input image.
        threshold (float): Threshold value for binarization.
        sigma (float): Standard deviation for Gaussian filter smoothing.
    
    Returns:
        binary_image (np.ndarray): Binary image after thresholding.
    """
    i_image = gaussian_filter(i_image.astype(float), sigma=sigma)
    binary_image = (i_image < threshold).astype(float)
    return binary_image


def generate_cluster_labels(i_image, low_threshold, high_threshold):
    """
    Generate cluster labels from variable data using thresholds and clustering.

    Parameters:
        i_image (np.ndarray): Input image representing any chosen variable.
        low_threshold (float): Lower threshold for ensemble detection.
        high_threshold (float): Higher threshold for core detection.
    
    Returns:
        labeled_ensemble_image (np.ndarray): Labels for ensemble clusters.
        labeled_core_image (np.ndarray): Labels for core clusters.
        labeled_envelop_image (np.ndarray): Labels for the envelope of clusters.
        labeled_total (np.ndarray): Combined labels of cores and envelopes.
    """
    low_threshold_binary_image = create_binary_image(i_image, threshold=low_threshold)
    high_threshold_binary_image = create_binary_image(i_image, threshold=high_threshold)
    
    ensemble_idx = np.array(np.where(low_threshold_binary_image)).T

    if ensemble_idx.size == 0:
        print("No ensemble points found.")
        return (np.zeros_like(i_image),) * 3

    # Distance matrix and clustering for ensemble
    distance_matrix_ensemble = pairwise_distances(ensemble_idx, metric=periodic_distance)
    clustering_ensemble = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.1, linkage='single', affinity='precomputed'
    ).fit(distance_matrix_ensemble)

    ######depends on python version................. 
    #clustering_ensemble = AgglomerativeClustering(
    #    n_clusters=None, distance_threshold=1.1, linkage='single', metric='precomputed'
    #).fit(distance_matrix_ensemble)
    
    labeled_ensemble_image = np.zeros_like(i_image)
    labeled_core_image = np.zeros_like(i_image)
    labeled_ensemble_image[np.where(low_threshold_binary_image)] = clustering_ensemble.labels_ + 1
    
    start_label_core = 1
    labeled_envelop_image = np.zeros_like(i_image)
    
    for label_ensemble in np.unique(labeled_ensemble_image):
        idx_core_ensemble = np.where((labeled_ensemble_image == label_ensemble) & (high_threshold_binary_image == 1))
        idx_core_ensemble_reshape = np.array(idx_core_ensemble).T

        idx_ensemble_label = np.where(labeled_ensemble_image == label_ensemble)
        
        if idx_core_ensemble_reshape.size <= 2:
            labeled_ensemble_image[idx_ensemble_label] = 0
            labeled_core_image[idx_ensemble_label] = 0
            labeled_envelop_image[idx_ensemble_label] = 0
        else:
            # Core clustering
            distance_matrix_core = pairwise_distances(idx_core_ensemble_reshape, metric=periodic_distance)
            clustering_core_ensemble = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.1, linkage='single', affinity='precomputed'
            ).fit(distance_matrix_core)

            #clustering_core_ensemble = AgglomerativeClustering(
            #    n_clusters=None, distance_threshold=1.1, linkage='single', metric='precomputed'
            #).fit(distance_matrix_core)
            labeled_core_image[idx_core_ensemble] = clustering_core_ensemble.labels_ + start_label_core
            
            start_label_core += np.max(clustering_core_ensemble.labels_) + 1

            # Envelope labeling with k-NN
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
    """Measure intersection between two labeled images."""
    intersection_indices = np.where((image1 == label1) & (image2 == label2))
    intersection_count = len(intersection_indices[0])
    total_label1_count = len(np.where(image1 == label1)[0])
    return intersection_count / total_label1_count



def track_clusters_over_time(list_labeled_image, similarity_threshold):
    image_1 = list_labeled_image[0]
    list_new_seq = [image_1]

    #plt.imshow(image_1%10, vmin=0, vmax=10)
    #plt.title('image2')
    #plt.colorbar()
    #plt.show()

    #for i_image in range(1,len(env_sequence)):

    for i_image in range(1, len(list_labeled_image)):
        #print(f'#######new_turn {i_image}')
        max_label_step_t = np.max(image_1)+10000
        
        image_2 = list_labeled_image[i_image]+max_label_step_t
        image_2[image_2==max_label_step_t] = 0

        list_labels_1 = list(np.unique(image_1))
        list_labels_1.remove(0)
        list_labels_2 = list(np.unique(image_2))
        
        list_labels_2.remove(0)
        
        

        if ((len(list_labels_1)==0) or (len(list_labels_2)==0)):
            output_image_step_tt = image_2

        else:
            similarity_matrix = np.zeros((len(list_labels_1), len(list_labels_2)))
            output_image_step_tt = copy.deepcopy(image_2)

            for i, label1 in enumerate(list_labels_1):
                for j,label2 in enumerate(list_labels_2):
                    similarity = measure_intersection(image1=image_1, label1=label1, image2=image_2,label2=label2)
                    similarity_matrix[i,j] = similarity

            #print(similarity_matrix)
            #print(np.argmax(similarity_matrix, axis=0))


            for j,label2 in enumerate(list_labels_2):
                new_label_j = list_labels_1[np.argmax(similarity_matrix, axis=0)[j]]
                assert new_label_j!=0
                if np.max(similarity_matrix, axis=0)[j]>similarity_threshold:
                    #print(label2, '--->', new_label_j)
                    output_image_step_tt[image_2==label2] = new_label_j
                else:
                    pass

        #plt.imshow(output_image_step_tt%10, vmin=0, vmax=10)
        #plt.title('image2')
        #plt.colorbar()
        #plt.show()
        #print('nb_out', len(np.unique(output_image_step_tt)))
        list_new_seq.append(output_image_step_tt)
        image_1 = output_image_step_tt
        
   
    
    
    #return np.array(list_new_seq)
    input_array = np.array(list_new_seq)
    relabeled_labeled_image_seq = np.zeros_like(input_array)
    for i,label in enumerate(np.unique(input_array)): #np.unique is always sorted such that label=0 is i=0
        relabeled_labeled_image_seq[input_array==label] = i
        
        
    return relabeled_labeled_image_seq
