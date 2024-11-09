#!/bin/sh
 
#SBATCH -A glab                  # Replace ACCOUNT with your account name
#SBATCH -N 8
#SBATCH -c 2
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-10:30            # Runtime in D-HH:MM
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=sga2133@columbia.edu
#SBATCH --mem=4G # Set the memory limit to 4GB

# Load necessary modules
module load netcdf-fortran/4.5.3
module load openmpi/gcc/64/4.1.5a1


# Print the current date and time
echo "Job started on $(date)"
# Read folder name from the file
foldername=$(cat OutputDirName)
# Navigate to the working directory


cd /burg/glab_new/users/sga2133/SAM_simulation_storage/$foldername/WORK
#cd /burg/home/sga2133/SAM_simulation/SAM6.11.8/$foldername/WORK


# Print the hostname and current directory
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

ls -l

# Set up debugging and error handling
set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as errors

# Print some information for debugging
echo "Running the MPI job..."

# Run the MPI job and redirect stdout and stderr to a log file
# mpiexec SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM > job_output.log 2>&1

chmod +x SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

mpiexec SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

# Print a completion message
#echo "MPI job completed on $(date)"


# Print the end date and time
echo "Job completed on $(date)"
