import os
import awswrangler as wr

def generate_simulation_paths(
        velocity: str, 
        temperature: str, 
        bowen_ratio: str, 
        microphysic: str, 
        s3_base_path: str = 's3://sam-simulations', 
        local_base_folder: str = '/home/ec2-user/DeepCloudLab/data') -> dict:
    """
    Generate the paths to the 3D, 2D, and 1D datasets and additional processed datasets based on simulation parameters.
    
    Parameters:
    - velocity (str): Velocity value, e.g., '0'.
    - temperature (str): Temperature value, e.g., '300'.
    - bowen_ratio (str): Bowen ratio value, e.g., '1'.
    - microphysic (str): Microphysics option, e.g., '1'.
    - s3_base_path (str): Base S3 path where the simulation data is stored.
    - local_base_folder (str): Local base directory where datasets will be saved per simulation.
    
    Returns:
    - dict: Dictionary containing S3 and local paths for 'output_data' and 'process_classes'.
    """

    output_dir='outputs'
    process_classes_dir='processed_classes'
    # Define the folder name based on simulation parameters
    folder_name = f'RCE_T{temperature}_U{velocity}_B{bowen_ratio}_M{microphysic}'
    folder_name_processed_classes = f'simulation_SAM_RCE_V{velocity}_T{temperature}_B{bowen_ratio}_M{microphysic}'


    # Base simulation paths for S3 and local storage
    s3_simulation_path_output = os.path.join(s3_base_path, output_dir, folder_name, 'WORK/NETCDF_files')
    local_simulation_path_output = os.path.join(local_base_folder, folder_name, 'WORK/NETCDF_files')
    
    # Base simulation paths for S3 and local storage
    s3_simulation_path_processes_classes = os.path.join(s3_base_path, process_classes_dir,  folder_name_processed_classes)
    local_simulation_path_processes_classes = os.path.join(local_base_folder, folder_name)
    

    # Construct paths for 3D, 2D, and 1D datasets (local and S3)
    file_3d = '3D/dataset_3d.nc'
    file_2d = f'2D/RCE_T{temperature}_U{velocity}_SAM{microphysic}MOM_B{bowen_ratio}_128x128x64_64.2Dcom_1.nc'
    file_1d = f'1D/RCE_T{temperature}_U{velocity}_SAM{microphysic}MOM_B{bowen_ratio}_128x128x64.nc'

    # S3 file paths for core datasets
    s3_path_3d = os.path.join(s3_simulation_path_output, file_3d)
    s3_path_2d = os.path.join(s3_simulation_path_output, file_2d)
    s3_path_1d = os.path.join(s3_simulation_path_output, file_1d)
    
    # Local file paths for core datasets
    local_path_3d = os.path.join(local_simulation_path_output, file_3d)
    local_path_2d = os.path.join(local_simulation_path_output, file_2d)
    local_path_1d = os.path.join(local_simulation_path_output, file_1d)

    # Additional datasets (processed classes)
    file_computed_2d = 'dataset_computed_2d'
    file_isentropic = 'dataset_isentropic'
    file_pickle_attributes = 'pickle_attributes'
    file_computed_3d = 'dataset_computed_3d'

    # S3 paths for additional datasets
    s3_path_computed_2d = os.path.join(s3_simulation_path_processes_classes, file_computed_2d)
    s3_path_isentropic = os.path.join(s3_simulation_path_processes_classes, file_isentropic)
    s3_path_pickle_attributes = os.path.join(s3_simulation_path_processes_classes, file_pickle_attributes)
    s3_path_computed_3d = os.path.join(s3_simulation_path_processes_classes, file_computed_3d)

    # Local paths for additional datasets
    local_path_computed_2d = os.path.join(local_simulation_path_processes_classes, file_computed_2d)
    local_path_isentropic = os.path.join(local_simulation_path_processes_classes, file_isentropic)
    local_path_pickle_attributes = os.path.join(local_simulation_path_processes_classes, file_pickle_attributes)
    local_path_computed_3d = os.path.join(local_simulation_path_processes_classes, file_computed_3d)
    
    # Return the structured dictionary with paths for output_data and processed_classes
    return {
        's3_paths': {
            'output_data': {
                'path_3d': s3_path_3d,
                'path_2d': s3_path_2d,
                'path_1d': s3_path_1d
            },
            'processed_classes': {
                'dataset_computed_2d': s3_path_computed_2d,
                'dataset_isentropic': s3_path_isentropic,
                'pickle_attributes': s3_path_pickle_attributes,
                'dataset_computed_3d': s3_path_computed_3d
            }
        },
        'local_paths': {
            'output_data': {
                'path_3d': local_path_3d,
                'path_2d': local_path_2d,
                'path_1d': local_path_1d
            },
            'processed_classes': {
                'dataset_computed_2d': local_path_computed_2d,
                'dataset_isentropic': local_path_isentropic,
                'pickle_attributes': local_path_pickle_attributes,
                'dataset_computed_3d': local_path_computed_3d
            }
        }
    }

def download_simulation_files(paths: dict):
    """
    Download all simulation dataset files from S3 to local storage if they do not already exist.
    
    Parameters:
    - paths (dict): Dictionary containing 's3_paths' and 'local_paths' for all datasets (both output_data and processes_classes).
    """
    # Extract s3 and local paths for output_data and processes_classes
    s3_paths = {**paths['s3_paths']['output_data'], **paths['s3_paths']['processed_classes']}
    local_paths = {**paths['local_paths']['output_data'], **paths['local_paths']['processed_classes']}

    # Create local directories if they don't exist
    for local_path in local_paths.values():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Function to download from S3 to local path if the file doesn't exist
    def download_from_s3(s3_path, local_path):
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'wb') as local_f:
                    wr.s3.download(path=s3_path, local_file=local_f)
                print(f"Downloaded {s3_path} to {local_path}")
            except Exception as e:
                print(f"Error downloading {s3_path}: {e}")
        else:
            print(f"{local_path} already exists, skipping download.")

    # Download all datasets from S3 to local paths if they don't exist
    for s3_path, local_path in zip(s3_paths.values(), local_paths.values()):
        download_from_s3(s3_path, local_path)
