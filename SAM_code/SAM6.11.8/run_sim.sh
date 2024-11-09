#!/bin/bash

# Define the directory containing the setup files
SETUP_DIR="/burg/home/sga2133/SAM_simulation/SAM6.11.8/setup_files"


# Loop through each set_up_file.txt in the setup_files folder
for setup_file in "$SETUP_DIR"/*.txt; do
    echo "Executing script with $setup_file"
    ./main_script.sh "$setup_file"
done
