#!/bin/sh
 
#SBATCH -A glab                  # Replace ACCOUNT with your account name
#SBATCH -N 8
#SBATCH -c 2
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-10:30            # Runtime in D-HH:MM
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=sga2133@columbia.edu
#SBATCH --mem=4G # Set the memory limit to 4GB

# Load modules
module load netcdf-fortran/4.5.3
module load openmpi/gcc/64/4.1.5a1
conda activate samenv

# Check if setup file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_setupfile.txt>"
    exit 1
fi

setup_file="$1"

# Check if the setup file exists
if [ ! -f "$setup_file" ]; then
    echo "Error: File $setup_file does not exist."
    exit 1
fi

# Read lines from setup file
mapfile -t lines < "$setup_file"

# Use the second line as the folder name
foldername="${lines[1]}"

# Set up paths
UtilDir=/burg/home/sga2133/SAM_simulation/SAM6.11.8/UTIL
path="/burg/glab_new/users/sga2133/SAM_simulation_storage/$foldername/WORK"

echo "**** $foldername"
echo "!!!! $path"



cd $path/NETCDF_files/3D
ncrcat RCE*.nc dataset_3d.nc


# Script end
