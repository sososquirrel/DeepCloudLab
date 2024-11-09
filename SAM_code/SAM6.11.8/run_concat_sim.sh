#!/bin/bash

# Define the directory containing the setup files
SETUP_DIR="/burg/home/sga2133/SAM_simulation/SAM6.11.8/setup_files"

# Check if the directory exists
if [ ! -d "$SETUP_DIR" ]; then
    echo "Error: Directory $SETUP_DIR does not exist."
    exit 1
fi

# Check if the post_process script exists and is executable
if [ ! -x "./create_sim_netcdf.sh" ]; then
    echo "Error: Script create_sim_netcdf.sh does not exist or is not executable."
    exit 1
fi

# Loop through each set_up_file.txt in the setup_files folder
for setup_file in "$SETUP_DIR"/*.txt; do
    if [ -f "$setup_file" ]; then
        echo "Executing post_process.sh with $setup_file"
        sbatch concat_3d_netcdf.sh "$setup_file"
    else
        echo "Skipping $setup_file as it is not a valid file."
    fi
done
