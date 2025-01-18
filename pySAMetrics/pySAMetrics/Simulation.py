"""Simulation class, allows to post process outputs of SAM"""

import pickle
from functools import partial
import numpy as np
import xarray as xr

#from pySAMetrics.cape.cape_functions import get_parcel_ascent
import pySAMetrics
from pySAMetrics import config

from pySAMetrics.relative_humidity import get_qsatt
from pySAMetrics.condensation_rate import get_condensation_rate

from pySAMetrics.cape.cape_functions import get_parcel_ascent

from pySAMetrics.coldpool_tracking import generate_cluster_labels, track_clusters_over_time
import os

from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import concurrent.futures

from pySAMetrics.diagnotic_fmse import diagnostic_fmse_z

from pySAMetrics.composite import instant_mean_extraction_data_over_extreme


#from pySAMetrics.cold_pool.composite import instant_mean_extraction_data_over_extreme

from pySAMetrics.utils import color, color2, distribution_tail, make_parallel, mass_flux, expand_array_to_tzyx_array
import dask

class Simulation:

    """Simulation creates an object that gathers all the datasets from simulation.
    It is designed here to study the convective response different parameters variations.
    An object Simulation represents 3Gb of data, and default settings load all the datasets.
    Everything you calculated is saved in a either a pickle or within a .nc dataset. 
    Methods save and load provide access to these previous calculus.
    You have a "plot mode" that allows you to partially load dataset and get access to your previous calculus.

    For plotting, a specify color is associated with each case. This might be very useful to keep consistence between plots.

    Attributes:
        color (cmap): matplotlib color defined for each case
        dataset_1d (xr.Datastet): Dataset with all 1d variables from SAM
        dataset_2d (xr.Datastet): Dataset with all 2d variables from SAM
        dataset_3d (xr.Datastet): Dataset with all 3d variables from SAM
        ...

    """

    def __init__(
        self,
        data_folder_paths: list,
        velocity: str,
        temperature:str,
        bowen_ratio:str,
        microphysic:str,
        plot_mode: bool = False,
    ):
        """Init"""
        # print("COUCOU SOSO ! CLASSE CETTE CLASSE !")

        self.velocity = velocity
        self.temperature = temperature
        self.bowen_ratio = bowen_ratio
        self.microphysic = microphysic

        self.name = f'RCE_T{self.temperature}_U{self.velocity}_B{self.bowen_ratio}_M{self.microphysic}'
    

        self.data_folder_paths = data_folder_paths


        self.dataset_1d = xr.open_dataset(self.data_folder_paths[0])#, decode_cf=False, engine='scipy')
        self.dataset_2d = xr.open_dataset(self.data_folder_paths[1])#, decode_cf=False, engine='scipy')
        self.dataset_3d = xr.open_dataset(self.data_folder_paths[2])#, decode_cf=False, engine='scipy')

        # Convert values to floats once for proper comparison
        bowen_ratio = float(self.bowen_ratio)
        temperature = float(self.temperature)
        velocity = float(self.velocity)

        # Handle Bowen Ratio
        if bowen_ratio != 1:
            cmap = pySAMetrics.cmap_simu_bowen
            alpha = (1 - bowen_ratio) / 1.3
            alpha = max(0, min(1, alpha))  # Ensure alpha is between 0 and 1
            self.color = cmap(alpha)

        # Handle Temperature
        elif temperature != 300:
            cmap = pySAMetrics.cmap_simu_temp
            alpha = (6 + (300 - temperature)) / 12
            alpha = max(0, min(1, alpha))  # Ensure alpha is between 0 and 1
            self.color = cmap(alpha)

        # Handle Velocity
        elif velocity != 0:
            cmap = pySAMetrics.cmap_simu_shear
            if velocity == 2.5:
                alpha = 0.2
            elif velocity == 5:
                alpha = 0.4
            elif velocity == 10:
                alpha = 0.7
            elif velocity == 20:
                alpha = 0.9
            else:
                alpha = 0  # Default alpha if velocity is unexpected
            self.color = cmap(alpha)

        # Default case
        else:
            self.color = 'k'

            


        #self.color=None #### define with organization or idk..

        self.time = self.dataset_3d.time.values
        self.X = self.dataset_3d.x.values
        self.Y = self.dataset_3d.y.values
        self.Z = self.dataset_3d.z.values

        sizes_data = self.dataset_3d.sizes
        self.nt, self.nz, self.nx, self.ny = sizes_data['time'], sizes_data['z'], sizes_data['x'], sizes_data['y']

        


    def save(self, backup_folder_path):
        """Save current instances of the class except starting datasets

        Args:
            backup_folder_path (str): path to the saving file
        """
        your_blacklisted_set = [
            "dataset_1d",
            "dataset_2d",
            "dataset_3d",
            "dataset_computed_3d",
            "dataset_computed_2d",
            "dataset_isentropic",
        ]
        dict2 = [
            (key, value)
            for (key, value) in self.__dict__.items()
            if key not in your_blacklisted_set
        ]


        simu_path = os.path.join(
            backup_folder_path,
            f"{self.name}"
        )
        # Construct the full file path
        attribute_file_path = os.path.join(
            backup_folder_path,
            f"{self.name}", "pickle_attributes"
        )
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(attribute_file_path), exist_ok=True)

        # Save the data to the file
        with open(attribute_file_path, "wb") as file:
            pickle.dump(dict2, file, 2)
        
        
        save_or_update_dataset(new_dataset=self.dataset_computed_2d, file_path=os.path.join(simu_path, 'dataset_computed_2d'))
        save_or_update_dataset(new_dataset=self.dataset_computed_3d, file_path=os.path.join(simu_path, 'dataset_computed_3d'))
        save_or_update_dataset(new_dataset=self.dataset_isentropic, file_path=os.path.join(simu_path, 'dataset_isentropic'))


    def load(self, backup_folder_path, load_all: bool = True):
        """Load calculated attributes from pickle backup files.

        Args:
            backup_folder_path (str): path to saved file
        """

        # Helper function to load a dataset
        def load_dataset(dataset_name):
            dataset_path = os.path.join(simu_path, dataset_name)
            if os.path.exists(dataset_path):
                try:
                    return xr.open_dataset(dataset_path, engine='netcdf4')
                except Exception as e:
                    print(f"A No dataset found: {dataset_name}")
            else:
                print(f"B No dataset found: {dataset_name}")
            return None

        # Construct the full file path
        attribute_file_path = os.path.join(backup_folder_path, f"{self.name}", "pickle_attributes")

        file = open(attribute_file_path,"rb")
        tmp_dict = pickle.load(file)
        file.close()

        # Check if the loaded object is a list and convert it to a dictionary if necessary
        if isinstance(tmp_dict, list):
            # Convert list of tuples to dict (assuming it is a list of key-value pairs)
            tmp_dict = dict(tmp_dict)

        # Check if the loaded object is a dictionary
        if not isinstance(tmp_dict, dict):
            raise ValueError(f"Expected a dictionary but got {type(tmp_dict)}")

        # Filter out blacklisted variables
        your_blacklisted_set = ['name'] if load_all else ['name']  # Add more variables to exclude if needed.
        tmp_dict = {key: value for key, value in tmp_dict.items() if key not in your_blacklisted_set}
        self.__dict__.update(tmp_dict)
        
        self.__dict__.update(tmp_dict)

        # Construct simulation path
        simu_path = os.path.join(backup_folder_path, self.name)

        print(f'****{simu_path}')

        # Load datasets

        self.dataset_isentropic = load_dataset('dataset_isentropic')
        self.dataset_computed_3d = load_dataset('dataset_computed_3d')
        self.dataset_computed_2d = load_dataset('dataset_computed_2d')

    def set_basic_variables_from_dataset(self):
        """Compute basic variables from dataset variable, adapted to (nt,nz,ny,nx) data."""

        # Cache values to avoid repeated access
        tabs_values = self.dataset_3d.TABS.values
        qv_values = self.dataset_3d.QV.values/1000 #in kg/kg
        p_values = self.dataset_1d.p.values[:self.nz]

        z_3d_in_time = pySAMetrics.utils.expand_array_to_tzyx_array(
            time_dependence=False,
            input_array=self.Z,
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        pressure_3d_in_time = pySAMetrics.utils.expand_array_to_tzyx_array(
            time_dependence=False,
            input_array=p_values,
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        FMSE = (
            pySAMetrics.HEAT_CAPACITY_AIR * tabs_values
            + pySAMetrics.GRAVITY * z_3d_in_time
            + pySAMetrics.LATENT_HEAT_AIR * qv_values
        )

        VIRTUAL_TEMPERATURE = tabs_values * (
            1 + (pySAMetrics.MIXING_RATIO_AIR_WATER_VAPOR - 1)
            / pySAMetrics.MIXING_RATIO_AIR_WATER_VAPOR * qv_values
        )

        vertical_mean_virtual_temperature_3d_in_time = pySAMetrics.utils.expand_array_to_tzyx_array(
            time_dependence=True,
            input_array=np.mean(VIRTUAL_TEMPERATURE, axis=(2, 3)),
            final_shape=np.array((self.nt, self.nz, self.ny, self.nx)),
        )

        POTENTIAL_TEMPERATURE = (
            tabs_values
            * (pySAMetrics.STANDARD_REFERENCE_PRESSURE / pressure_3d_in_time)
            ** pySAMetrics.GAS_CONSTANT_OVER_HEAT_CAPACITY_AIR
        )

        BUOYANCY = (
            pySAMetrics.GRAVITY
            * (VIRTUAL_TEMPERATURE - vertical_mean_virtual_temperature_3d_in_time)
            / vertical_mean_virtual_temperature_3d_in_time
        )

        VORTICITY = np.gradient(self.dataset_3d.U.values, self.Z, axis=1) - np.gradient(
            self.dataset_3d.W.values, self.X, axis=3
        )

        QSATT = get_qsatt(p_values, tabs_values)

        RH = qv_values / QSATT

        ###commenting CR for the moment
        #CR = get_condensation_rate(
        #    vertical_velocity=self.dataset_3d.W.values,
        #    density=self.dataset_1d.RHO.values,
        #    humidity=qv_values,
        #)
        
        RHO_W = pySAMetrics.utils.expand_array_to_tzyx_array(input_array = self.dataset_1d.RHO.values[-self.nt:],
                                                     final_shape = (self.nt,self.nz,self.nx,self.ny),
                                                     time_dependence=True)*self.dataset_3d.W.values
        
        self.dataset_computed_3d = xr.Dataset({
            'FMSE': (('time', 'z', 'y', 'x'), FMSE, {
                'long_name': 'Free Moist Static Energy',
                'units': 'J kg^-1'
            }),
            'VIRTUAL_TEMPERATURE': (('time', 'z', 'y', 'x'), VIRTUAL_TEMPERATURE, {
                'long_name': 'Virtual Temperature',
                'units': 'K'
            }),
            'POTENTIAL_TEMPERATURE': (('time', 'z', 'y', 'x'), POTENTIAL_TEMPERATURE, {
                'long_name': 'Potential Temperature',
                'units': 'K'
            }),
            'BUOYANCY': (('time', 'z', 'y', 'x'), BUOYANCY, {
                'long_name': 'Buoyancy',
                'units': 'm s^-2'
            }),
            'VORTICITY': (('time', 'z', 'y', 'x'), VORTICITY, {
                'long_name': 'Vorticity',
                'units': 's^-1'
            }),
            'RHO_W': (('time', 'z', 'y', 'x'), RHO_W, {
                'long_name': 'Vertical Mass Flux',
                'units': 'kg.m/s'
            })
        })

        
        self.dataset_computed_2d = xr.Dataset({
            'CR': (('time','y', 'x'), np.zeros_like(tabs_values[:,0,:,:]), {
                'long_name': 'Condensation rate',
                'units': 'mm'
            })
        })
        


    def set_CR_3D(self):
        self.CR_3D = get_condensation_rate(
            vertical_velocity=self.W.values,
            density=self.dataset_1d.RHO.values,
            humidity=self.dataset_3d.QV.values,
            return_3D=True,
        )

    def set_composite_variables(
        self,
        data_name: str,
        variable_to_look_for_extreme: str,
        extreme_events_choice: str,
        x_margin: int,
        y_margin: int,
        parallelize: bool = False,
        return_3D: bool = False,
        dataset_for_variable_2d: str = "dataset_2d",
        dataset_for_variable_3d: str = "dataset_3d",
        return_1D: bool = False,
    ) -> np.array:
        


        """Compute the composite, namely the conditionnal mean, of 2d or 3d variables evolving in time
        This method builds attribute

        Args:
            data_name (str): name of the variable composite method is applying to
            variable_to_look_for_extreme (str): name of the variable that describe extreme event
            extreme_events_choice (str): max 1-percentile or 10-percentile
            x_margin (int): width of window zoom
            y_margin (int, optional): depth of window zoom
            parallelize (bool, optional): use all your cpu power
        """

        if dataset_for_variable_3d == "":
            data_3d = getattr(self, data_name)
        else:
            data_3d = getattr(getattr(self, dataset_for_variable_3d), data_name)
            

        if dataset_for_variable_2d == "":
            data_2d = getattr(self, variable_to_look_for_extreme)
        else:
            data_2d = getattr(
                getattr(self, dataset_for_variable_2d), variable_to_look_for_extreme
            )

        if type(data_2d) == xr.core.dataarray.DataArray:
            data_2d = data_2d.values

        if type(data_3d) == xr.core.dataarray.DataArray:
            data_3d = data_3d.values

        data_2d = np.array(data_2d, dtype=np.float32)
        data_3d = np.array(data_3d, dtype=np.float32)


        if parallelize:
            print(pySAMetrics.N_CPU)
            parallel_composite = make_parallel(
                function=instant_mean_extraction_data_over_extreme, nprocesses=pySAMetrics.N_CPU
            )
            composite_variable = parallel_composite(
                iterable_values_1=data_3d,
                iterable_values_2=data_2d,
                extreme_events_choice=extreme_events_choice,
                x_margin=x_margin,
                y_margin=y_margin,
                return_3D=return_3D,
            )

        else:  # NO PARALLELIZATION
            composite_variable = []
            data = data_3d
            for image, variable_extreme in tqdm(zip(data, data_2d), total=len(data)):
                composite_variable.append(
                    instant_mean_extraction_data_over_extreme(
                        data=image,
                        variable_to_look_for_extreme=variable_extreme,
                        extreme_events_choice=extreme_events_choice,
                        x_margin=x_margin,
                        y_margin=y_margin,
                        return_3D=return_3D,
                    )
                )

        composite_variable = np.array(composite_variable)
        composite_variable = np.mean(composite_variable, axis=0)


        if return_3D:
            setattr(
                self,
                data_name + "_composite_" + variable_to_look_for_extreme + "_3D",
                composite_variable,
            )
        else:
            setattr(
                self,
                f'{data_name}_composite_{variable_to_look_for_extreme}',
                composite_variable,
            )
        if return_1D:
            if return_3D:
                setattr(
                    self,
                    data_name + "_composite_" + variable_to_look_for_extreme + "_1D",
                    composite_variable[:, x_margin, y_margin],
                )
            else:
                setattr(
                    self,
                    data_name + "_composite_" + variable_to_look_for_extreme + "_1D",
                    composite_variable[:, x_margin],
                )



    def get_cape(
        self,
        temperature: str = "TABS",
        vertical_array: str = "z",
        pressure: str = "p",
        humidity_ground: str = "QV",
        parallelize: bool = True,
        set_parcel_ascent_composite_1d: bool = True,
    ) -> np.array:

        # get the variable in np.array
        temperature_array = getattr(self.dataset_3d, temperature).values

        nt, nz, ny, nx = temperature_array.shape

        z_array = getattr(self.dataset_3d, vertical_array).values
        pressure_array = getattr(self.dataset_1d, pressure).values[:nz]
        humidity_ground_array = getattr(self.dataset_3d, humidity_ground).values[:, 0, :, :]

        # calculate parcel_ascent
        if parallelize:
            parallel_parcel_ascent = make_parallel(
                function=get_parcel_ascent, nprocesses=config.N_CPU
            )
            parcel_ascent = parallel_parcel_ascent(
                iterable_values_1=temperature_array,
                iterable_values_2=humidity_ground_array / 1000,
                pressure=pressure_array,
                vertical_array=z_array,
            )

        else:
            parcel_ascent = []

            for temperature_i, humidity_ground_i in zip(
                temperature_array,
                humidity_ground_array / 1000,
            ):

                T_parcel_i = get_parcel_ascent(
                    temperature=temperature_i,
                    humidity_ground=humidity_ground_i,
                    pressure=pressure_array,
                    vertical_array=z_array,
                )

                parcel_ascent.append(T_parcel_i)

        parcel_ascent = np.array(parcel_ascent)

        dz = np.gradient(z_array)
        dz_3d = pySAMetrics.utils.expand_array_to_tzyx_array(
            input_array=dz, time_dependence=False, final_shape=temperature_array.shape
        )

        cape = pySAMetrics.GRAVITY * np.sum(
            dz_3d
            * pySAMetrics.utils.max_point_wise(
                matrix_1=np.zeros_like(parcel_ascent),
                matrix_2=((parcel_ascent - temperature_array) / temperature_array),
            ),
            axis=1,
        )

        #if set_parcel_ascent_composite_1d:
            #setattr(self, "parcel_ascent", parcel_ascent)

            #self.set_composite_variables(
                #data_name="parcel_ascent",
                #variable_to_look_for_extreme="cape",
                #extreme_events_choice="max",
                #x_margin=40,
                #y_margin=40,
                #dataset_for_variable_2d="",
                #dataset_for_variable_3d="",
                #return_1D=True,
            #)

            #delattr(self, "parcel_ascent")

                
        self.dataset_computed_2d = self.dataset_computed_2d.assign(
                    CAPE=(('time','y', 'x'), cape, {
                        'long_name': 'CAPE',
                        'units': 'J/kg/m2'
                    })
                )
                        
        self.dataset_computed_3d = self.dataset_computed_3d.assign(
                    PARCEL_ASCENT=(('time','z', 'y','x'), parcel_ascent, {
                        'long_name': 'Parcel ascent',
                        'units': 'J/kg'
                    })
                )


    
    def get_coldpool_tracking_images(self, variable_images, low_threshold, high_threshold):
        num_workers = 32  # Set number of workers based on your system
        

        # Create a partial function where the other arguments (variable_images, low_threshold, high_threshold) are pre-filled
        process_func = partial(process_variable_images, variable_images=variable_images, 
                               low_threshold=low_threshold, high_threshold=high_threshold)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            print(f"Number of workers: {num_workers}")
            # Use the pre-filled partial function with the time step as the argument
            results = list(tqdm(executor.map(process_func, range(self.nt)), total=self.nt))
            #results = list(tqdm(executor.map(process_func, range(10)), total=10)) for debug

        # Process results if needed
        results=np.array(results)

        core_sequence, envelop_sequence, labeled_total = results[:,0], results[:,1], results[:,2]

        core_sequence[core_sequence!=0]=1
        envelop_sequence[envelop_sequence!=0]=1

        self.label_total = labeled_total

        labeled_image_seq=[]
        labeled_image_seq.extend(labeled_total)
        SIMILARITY_THRESHOLD=0.55
        tracking_cp = track_clusters_over_time(list_labeled_image=labeled_image_seq, similarity_threshold=SIMILARITY_THRESHOLD)
        
        
        labeled_total[labeled_total!=0]=1

        # Check if self.dataset_2d exists
        if hasattr(self, 'dataset_computed_2d') and self.dataset_computed_2d is not None:
            # If self.dataset_computed_2d exists, assign the new variables
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                CORE_BINARY=(('time', 'y', 'x'), core_sequence, {
                    'long_name': 'Core Cold Pool location',
                    'units': '1'
                })
            )
            
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                ENVELOP_BINARY=(('time', 'y', 'x'), envelop_sequence, {
                    'long_name': 'Envelop Cold Pool location',
                    'units': '1'
                })
            )
            
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                CP_BINARY=(('time', 'y', 'x'), labeled_total, {
                    'long_name': 'Cold Pool location',
                    'units': '1'
                })
            )
            
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                CP_LABELS=(('time', 'y', 'x'), tracking_cp, {
                    'long_name': 'Cold Pool label',
                    'units': '1'
                })
            )
        else:
            # If self.dataset_computed_2d does not exist, create an empty dataset with coordinates
            self.dataset_computed_2d = xr.Dataset(
                coords={
                    'time': range(len(core_sequence)),  # or actual time values
                    'y': range(len(core_sequence[0])),   # or actual y values
                    'x': range(len(core_sequence[0][0]))  # or actual x values
                }
            )
            # Now assign the new variables
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                CORE_BINARY=(('time', 'y', 'x'), core_sequence, {
                    'long_name': 'Core Cold Pool location',
                    'units': '1'
                })
            )
            
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                ENVELOP_BINARY=(('time', 'y', 'x'), envelop_sequence, {
                    'long_name': 'Envelop Cold Pool location',
                    'units': '1'
                })
            )
            
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                CP_BINARY=(('time', 'y', 'x'), labeled_total, {
                    'long_name': 'Cold Pool location',
                    'units': '1'
                })
            )
            
            self.dataset_computed_2d = self.dataset_computed_2d.assign(
                CP_LABELS=(('time', 'y', 'x'), tracking_cp, {
                    'long_name': 'Cold Pool label',
                    'units': '1'
                })
            )

    def get_isentropic_dataset(self):
        # Extract arrays once to avoid redundancy
        fmse_array = self.dataset_computed_3d.FMSE.values
        z_array = self.dataset_3d.z.values
        rho_w_array = self.dataset_computed_3d.RHO_W.values
        qn_array = self.dataset_3d.QN.values
        qp_array = self.dataset_3d.QP.values

        # Create partial function with pre-filled arguments
        process_func = partial(
            process_time_step_sum, 
            fmse_array=fmse_array,
            z_array=z_array, 
            rho_w_array=rho_w_array, 
            qn_array=qn_array, 
            qp_array=qp_array
        )

        nt = self.nt  # Number of timesteps
       
        isent_rho_w_sum=[]
        isent_qn_sum=[]
        isent_qp_sum=[]
        for i_t in tqdm(range(self.nt)):
            results_i=(process_func(i_t))
            isent_rho_w_sum.append(results_i[0])
            isent_qn_sum.append(results_i[1])
            isent_qp_sum.append(results_i[2])
        
        # Create partial function with pre-filled arguments
        process_func = partial(
            process_time_step_mean, 
            fmse_array=fmse_array,
            z_array=z_array, 
            rho_w_array=rho_w_array, 
            qn_array=qn_array, 
            qp_array=qp_array
        )

        nt = self.nt  # Number of timesteps
       
        isent_rho_w_mean=[]
        isent_qn_mean=[]
        isent_qp_mean=[]
        for i_t in tqdm(range(self.nt)):
            results_i=(process_func(i_t))
            isent_rho_w_mean.append(results_i[0])
            isent_qn_mean.append(results_i[1])
            isent_qp_mean.append(results_i[2])

        # Create partial function with pre-filled arguments
        process_func = partial(
            process_time_step_max, 
            fmse_array=fmse_array,
            z_array=z_array, 
            rho_w_array=rho_w_array, 
            qn_array=qn_array, 
            qp_array=qp_array
        )

        nt = self.nt  # Number of timesteps
       
        isent_rho_w_max=[]
        isent_qn_max=[]
        isent_qp_max=[]
        for i_t in tqdm(range(self.nt)):
            results_i=(process_func(i_t))
            isent_rho_w_max.append(results_i[0])
            isent_qn_max.append(results_i[1])
            isent_qp_max.append(results_i[2])

        # Unpack results
        #isent_rho_w, isent_qn, isent_qp = np.zeros((3,3,3)), np.zeros((3,3,3)), np.zeros((3,3,3))

        # Create the final xarray Dataset
        self.dataset_isentropic = xr.Dataset({
            'RHO_W_sum': (('time', 'z', 'fmse'), isent_rho_w_sum, {
                'long_name': 'Vertical Mass Flux',
                'units': 'kg/m2.s'
            }),
            'QN_sum': (('time', 'z', 'fmse'), isent_qn_sum, {
                'long_name': 'Cloud Water',
                'units': 'g/kg'
            }),
            'QP_sum': (('time', 'z', 'fmse'), isent_qp_sum, {
                'long_name': 'Precipitating Water',
                'units': 'g/kg'
            }),
            'RHO_W_mean': (('time', 'z', 'fmse'), isent_rho_w_mean, {
                'long_name': 'Vertical Mass Flux',
                'units': 'kg/m2.s'
            }),
            'QN_mean': (('time', 'z', 'fmse'), isent_qn_mean, {
                'long_name': 'Cloud Water',
                'units': 'g/kg'
            }),
            'QP_mean': (('time', 'z', 'fmse'), isent_qp_mean, {
                'long_name': 'Precipitating Water',
                'units': 'g/kg'
            }),
            'RHO_W_max': (('time', 'z', 'fmse'), isent_rho_w_max, {
                'long_name': 'Vertical Mass Flux',
                'units': 'kg/m2.s'
            }),
            'QN_max': (('time', 'z', 'fmse'), isent_qn_max, {
                'long_name': 'Cloud Water',
                'units': 'g/kg'
            }),
            'QP_max': (('time', 'z', 'fmse'), isent_qp_max, {
                'long_name': 'Precipitating Water',
                'units': 'g/kg'
            }),
        })



def process_variable_images(time_step, variable_images, low_threshold, high_threshold):
    # This function should include the logic to process a specific time step of variable_images
    # Add your image processing code here using low_threshold, high_threshold, etc.
    # For example:
    labeled_core_image, labeled_envelop_image, labeled_total = generate_cluster_labels(
        variable_images[time_step, :, :], low_threshold, high_threshold
    )
    return labeled_core_image, labeled_envelop_image, labeled_total

# Move the process_time_step function outside of the method
def process_time_step_sum(i_t, fmse_array, z_array, rho_w_array, qn_array, qp_array):
    """Processes a single time step and returns the computed values."""
    isent_rho_w_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=rho_w_array,
        time_step=i_t,
        bin_mode='sum'
    )
    
    isent_qn_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=qn_array,
        time_step=i_t,
        bin_mode='sum'
    )
    
    isent_qp_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=qp_array,
        time_step=i_t,
        bin_mode='sum'
    )
    
    return isent_rho_w_i, isent_qn_i, isent_qp_i

def process_time_step_mean(i_t, fmse_array, z_array, rho_w_array, qn_array, qp_array):
    """Processes a single time step and returns the computed values."""
    isent_rho_w_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=rho_w_array,
        time_step=i_t,
        bin_mode='mean'
    )
    
    isent_qn_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=qn_array,
        time_step=i_t,
        bin_mode='mean'
    )
    
    isent_qp_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=qp_array,
        time_step=i_t,
        bin_mode='mean'
    )
    
    return isent_rho_w_i, isent_qn_i, isent_qp_i

def process_time_step_max(i_t, fmse_array, z_array, rho_w_array, qn_array, qp_array):
    """Processes a single time step and returns the computed values."""
    isent_rho_w_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=rho_w_array,
        time_step=i_t,
        bin_mode='max'
    )
    
    isent_qn_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=qn_array,
        time_step=i_t,
        bin_mode='max'
    )
    
    isent_qp_i = diagnostic_fmse_z(
        fmse_array=fmse_array,
        z_array=z_array,
        data_array=qp_array,
        time_step=i_t,
        bin_mode='max'
    )
    
    return isent_rho_w_i, isent_qn_i, isent_qp_i


def save_or_update_dataset(new_dataset: xr.Dataset, file_path: str):
    """
    Save a new dataset to a file or update the existing dataset by adding any new variables.
    
    Parameters:
    - new_dataset (xr.Dataset): The dataset containing new variables to add.
    - file_path (str): The path to the file where the dataset is stored.
    """
    if os.path.exists(file_path):
        # Load the existing dataset
        existing_dataset = xr.open_dataset(file_path)
        
        # Find variables that are in the new dataset but not in the existing one
        new_vars = [var for var in new_dataset.data_vars if var not in existing_dataset.data_vars]

        if new_vars:
            print(f"Adding new variables: {new_vars}")
            # Select only the new variables to add
            new_data_to_add = new_dataset[new_vars]
            
            # Merge the new variables into the existing dataset
            updated_dataset = xr.merge([existing_dataset, new_data_to_add])
            existing_dataset.close()
            
            # Save the updated dataset
            updated_dataset.to_netcdf(file_path, mode='w')
            print(f"Updated dataset saved with new variables: {new_vars}")
        else:
            print("No new variables to add. Dataset remains unchanged.")
    else:
        # If the file doesn't exist, save the new dataset directly
        new_dataset.to_netcdf(file_path)
        print(f"New dataset saved at {file_path}")