"""
Base Simulation class for post-processing outputs of SAM.
"""

# Standard Library Imports
import os
import pickle
import logging
from typing import List, Optional, Union, Dict, Any

# Third-Party Imports
import numpy as np
import xarray as xr

import tempfile
import shutil

# Module logging
logger = logging.getLogger(__name__)


class Simulation:
    """
    Base Simulation class that gathers all the datasets from a simulation.
    
    This class is designed to study the convective response to different parameter variations.
    An object Simulation represents ~3GB of data, and default settings load all the datasets.
    Calculations are saved either as pickle files or within .nc datasets. 
    Methods save and load provide access to these previous calculations.
    
    Attributes:
        color (cmap): matplotlib color defined for each case
        dataset_1d (xr.Dataset): Dataset with all 1d variables from SAM
        dataset_2d (xr.Dataset): Dataset with all 2d variables from SAM
        dataset_3d (xr.Dataset): Dataset with all 3d variables from SAM
        time (np.array): Time values from the dataset
        X, Y, Z (np.array): Spatial coordinates
        name (str): Name of the simulation based on parameters
    """

    def __init__(
        self,
        data_folder_paths: List[str],
        velocity: str,
        temperature: str,
        bowen_ratio: str,
        microphysic: str,
        split: Optional[int] = None,
        plot_mode: bool = False,
    ):
        """
        Initialize Simulation object.
        
        Args:
            data_folder_paths: List of paths to data folders
            velocity: Velocity parameter
            temperature: Temperature parameter
            bowen_ratio: Bowen ratio parameter
            microphysic: Microphysics parameter
            split: Split parameter (optional)
            plot_mode: Whether to load datasets in plot mode (partial loading)
        """
        self.velocity = velocity
        self.temperature = temperature
        self.bowen_ratio = bowen_ratio
        self.microphysic = microphysic
        self.split = split
        self.plot_mode = plot_mode
        self.data_folder_paths = data_folder_paths

        # Set simulation name
        if split is not None:
            self.name = f'RCE_T{self.temperature}_U{self.velocity}_B{self.bowen_ratio}_M{self.microphysic}_split_{self.split}'
        else:
            self.name = f'RCE_T{self.temperature}_U{self.velocity}_B{self.bowen_ratio}_M{self.microphysic}'
        
        self._load_datasets()
        self._initialize_dimensions()
        self._set_simulation_color()

    def _load_datasets(self) -> None:
        """Load the 1D, 2D, and 3D datasets."""
        try:
            if not self.plot_mode:
                self.dataset_1d = xr.open_dataset(self.data_folder_paths[0])
                self.dataset_2d = xr.open_dataset(self.data_folder_paths[1])
                self.dataset_3d = xr.open_dataset(self.data_folder_paths[2])
            else:
                # In plot mode, load minimal data or metadata
                logger.info("Loading in plot mode (minimal data)")
                # Implementation for plot mode could be added here
        except (IndexError, FileNotFoundError) as e:
            logger.error(f"Error loading datasets: {e}")
            raise

    def _initialize_dimensions(self) -> None:
        """Initialize time and spatial dimensions from the datasets."""
        self.time = self.dataset_3d.time.values
        self.X = self.dataset_3d.x.values
        self.Y = self.dataset_3d.y.values
        self.Z = self.dataset_3d.z.values

        sizes_data = self.dataset_3d.sizes
        self.nt = sizes_data.get('time', 0)
        self.nz = sizes_data.get('z', 0)
        self.nx = sizes_data.get('x', 0)
        self.ny = sizes_data.get('y', 0)

    def _set_simulation_color(self) -> None:
        """Set a color for the simulation based on parameters."""
        from pySAMetrics.utils_simulation import get_simulation_color
        
        self.color = get_simulation_color(
            bowen_ratio=self.bowen_ratio, 
            temperature=self.temperature, 
            velocity=self.velocity
        )

    def save(self, backup_folder_path: str, locking_h5=False) -> None:
        """
        Save current instances of the class except starting datasets.
        
        Args:
            backup_folder_path: Path to the directory where data will be saved
        """
        if locking_h5:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        # Define attributes that should not be saved
        blacklisted_attrs = [
            "dataset_1d",
            "dataset_2d",
            "dataset_3d",
            "dataset_computed_3d",
            "dataset_computed_2d",
            "dataset_isentropic",
        ]
        
        # Create dictionary of attributes to save
        attrs_to_save = [
            (key, value)
            for (key, value) in self.__dict__.items()
            if key not in blacklisted_attrs
        ]

        # Create simulation directory path
        simu_path = os.path.join(backup_folder_path, self.name)
        
        # Create attributes file path
        attribute_file_path = os.path.join(simu_path, "pickle_attributes")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(attribute_file_path), exist_ok=True)

        # Save attributes to pickle file
        with open(attribute_file_path, "wb") as file:
            pickle.dump(attrs_to_save, file, 2)
        
        # Save computed datasets if they exist
        self._save_dataset_if_exists('dataset_computed_2d', os.path.join(simu_path, 'dataset_computed_2d'))
        self._save_dataset_if_exists('dataset_computed_3d', os.path.join(simu_path, 'dataset_computed_3d'))
        self._save_dataset_if_exists('dataset_isentropic', os.path.join(simu_path, 'dataset_isentropic'))

    def _save_dataset_if_exists(self, attr_name: str, file_path: str) -> None:
        """Save a dataset if it exists as an attribute.
        
        Args:
            attr_name: Name of the attribute containing the dataset
            file_path: Path where to save the dataset
        """
        if hasattr(self, attr_name):
            dataset = getattr(self, attr_name)
            if dataset is not None:
                save_or_update_dataset(dataset, file_path)

    def load(self, backup_folder_path: str, load_all: bool = True) -> None:
        """
        Load calculated attributes from pickle backup files.

        Args:
            backup_folder_path: Path to the directory containing saved data
            load_all: Whether to load all attributes
        """
        # Construct simulation path
        simu_path = os.path.join(backup_folder_path, self.name)
        
        # Construct attributes file path
        attribute_file_path = os.path.join(simu_path, "pickle_attributes")
        
        try:
            # Load pickled attributes
            with open(attribute_file_path, "rb") as file:
                tmp_dict = pickle.load(file)
            
            # Convert list to dictionary if necessary
            if isinstance(tmp_dict, list):
                tmp_dict = dict(tmp_dict)
            
            # Ensure loaded object is a dictionary
            if not isinstance(tmp_dict, dict):
                raise ValueError(f"Expected a dictionary but got {type(tmp_dict)}")
            
            # Filter out blacklisted attributes
            blacklisted_attrs = ['name'] if load_all else ['name']
            tmp_dict = {k: v for k, v in tmp_dict.items() if k not in blacklisted_attrs}
            
            # Update instance attributes
            self.__dict__.update(tmp_dict)
            
            # Load datasets
            logger.info(f'Loading data from {simu_path}')
            self._load_dataset('dataset_isentropic', os.path.join(simu_path, 'dataset_isentropic'))
            self._load_dataset('dataset_computed_3d', os.path.join(simu_path, 'dataset_computed_3d'))
            self._load_dataset('dataset_computed_2d', os.path.join(simu_path, 'dataset_computed_2d'))
            
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _load_dataset(self, attr_name: str, file_path: str) -> None:
        """Load a dataset from file and set it as an attribute.
        
        Args:
            attr_name: Name of the attribute to set
            file_path: Path to the dataset file
        """
        if os.path.exists(file_path):
            try:
                dataset = xr.open_dataset(file_path, engine='netcdf4')
                setattr(self, attr_name, dataset)
            except Exception as e:
                logger.warning(f"Could not load dataset {attr_name}: {e}")
                setattr(self, attr_name, None)
        else:
            logger.warning(f"Dataset file not found: {file_path}")
            setattr(self, attr_name, None)




def save_or_update_dataset(new_dataset: xr.Dataset, file_path: str) -> None:
    """
    Save a new dataset or update an existing one with new variables in a safe, portable way.
    Works reliably on external drives without relying on file locking.
    """
    try:
        if os.path.exists(file_path):
            # Load existing dataset and check for new variables
            with xr.open_dataset(file_path) as existing_dataset:
                new_vars = [var for var in new_dataset.data_vars if var not in existing_dataset.data_vars]

                if new_vars:
                    logger.info(f"Adding new variables: {new_vars}")
                    updated_dataset = xr.merge([existing_dataset, new_dataset[new_vars]])
                else:
                    logger.info("No new variables to add. Dataset remains unchanged.")
                    return  # Nothing to do

            # Write to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(file_path)) as tmp:
                temp_path = tmp.name
            updated_dataset.to_netcdf(temp_path, mode='w')

            # Move to final destination
            shutil.move(temp_path, file_path)
            logger.info(f"Updated dataset written to {file_path}")
        else:
            new_dataset.to_netcdf(file_path)
            logger.info(f"New dataset saved at {file_path}")
    except Exception as e:
        logger.error(f"Error saving or updating dataset: {e}")