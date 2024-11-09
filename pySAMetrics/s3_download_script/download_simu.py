from utils_s3 import generate_simulation_paths, download_simulation_files
from tqdm import tqdm

simulations_dict = {
    'RCE_T295_U0_B1_M1': {
        'velocity': '0',
        'temperature': '295',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T300_U0_B1_M2': {
        'velocity': '0',
        'temperature': '300',
        'bowen_ratio': '1.2',
        'microphysic': '2'
    },
    'RCE_T300_U0_B0.1_M1': {
        'velocity': '0',
        'temperature': '300',
        'bowen_ratio': '0.1',
        'microphysic': '1'
    },
    'RCE_T300_U10_B1_M1': {
        'velocity': '10',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T300_U2.5_B1_M1': {
        'velocity': '2.5',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T302_U0_B1_M1': {
        'velocity': '0',
        'temperature': '302',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T300_U0_B0.5_M1': {
        'velocity': '0',
        'temperature': '300',
        'bowen_ratio': '0.5',
        'microphysic': '1'
    },
    'RCE_T300_U10_B1_M1': {
        'velocity': '10',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T300_U2.5_B1_M1': {
        'velocity': '2.5',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T305_U0_B1_M1': {
        'velocity': '0',
        'temperature': '305',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    #'RCE_T300_U0_B1_M1': {
    #    'velocity': '0',
    #    'temperature': '300',
    #    'bowen_ratio': '1',
    #    'microphysic': '1'
    #},
    'RCE_T300_U20_B1_M1': {
        'velocity': '20',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1'
    },
    'RCE_T300_U5_B1_M1': {
        'velocity': '5',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1'
    }
}


for simu_name, simu_data in tqdm(simulations_dict.items()):
    print(f"Simulation {simu_name} is being downloaded...")
    paths_dict = generate_simulation_paths(**simu_data)

    download_simulation_files(paths_dict)