#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=test_1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=1-4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --exclusive
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ss6287@columbia.edu

# Purge existing modules
module purge

# Load the available OpenMPI module
module load openmpi/gcc/64/4.1.1_cuda_11.0.3_aware

# Load the available Anaconda module
module load anaconda/3-2022.05

foldername=$(cat OutputDirName)
# Navigate to the working directory
cd /burg/home/sga2133/SAM_simulation/SAM6.11.8/$foldername/WORK

# Use mpiexec for MPI jobs
# Specify the full path to gmx_mpi if it's not in your $PATH
mpiexec SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

# End of script

