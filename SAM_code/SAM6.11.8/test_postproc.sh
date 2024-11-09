#!/bin/bash

#load modules
module load netcdf-fortran/4.5.3
module load openmpi/gcc/64/4.1.5a1

#OutputDirName=relaxSST300tau2_H50_ps_fR_new/
#OutputDirName=D576_isl80_dt5_cold_SST300_radhom/
#select RCE_290_D_576_ocean_fullrad 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR
UtilDir=/burg/home/sga2133/SAM_simulation/SAM6.11.8/UTIL
foldername=$(cat OutputDirName)

echo "**** $foldername"

path=/burg/home/sga2133/SAM_simulation/SAM6.11.8/$foldername/WORK

mkdir /burg/glab/users/sga2133/SAM_simulation_storage/$foldername

path_store=/burg/glab/users/sga2133/SAM_simulation_storage/$foldername


echo "!!!! $path_store"

mkdir $path_store/NETCDF_files
mkdir $path_store/NETCDF_files/1D
mkdir $path_store/NETCDF_files/2D
mkdir $path_store/NETCDF_files/3D

