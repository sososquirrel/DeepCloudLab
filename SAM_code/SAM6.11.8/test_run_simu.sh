#!/bin/bash

# Paths to the main script and the base setup file
base_setup_file="set_up.txt"

# Define the control parameters
control_surface_temp="300"
control_basal_velocity="2.5"
control_beta_coefficient="1"
control_microphysic_mode="1"

# Define the ranges for each parameter
surface_temp_variations=("290" "310")
basal_velocity_variations=("1" "5" "10" "15")
beta_coefficient_variations=("0.5" "1.5" "2")
microphysic_modes=("1" "2")

# Create a directory to store all setup files
setup_dir="setup_files"
mkdir -p "$setup_dir"

# Create and run the control simulation
control_output_dir="RCE_T${control_surface_temp}_${control_basal_velocity}_${control_beta_coefficient}_${control_microphysic_mode}_control"
control_setup_file="${setup_dir}/control_set_up.txt"
cp "$base_setup_file" "$control_setup_file"

# Modify the setup file for the control simulation
sed -i "2s/.*/${control_output_dir}/" "$control_setup_file"  # output_dir_name
sed -i "3s/.*/${control_surface_temp}/" "$control_setup_file"  # surface_temp
sed -i "4s/.*/${control_basal_velocity}/" "$control_setup_file"  # basal_velocity
sed -i "5s/.*/${control_beta_coefficient}/" "$control_setup_file"  # beta_coefficient
sed -i "6s/.*/${control_microphysic_mode}/" "$control_setup_file"  # microphysic_mode

echo "************************"

# Run the control simulation
./main_script.sh "$control_setup_file"

# Run variations by changing one parameter at a time

# 1. Varying surface temperature
for surface_temp in "${surface_temp_variations[@]}"; do
    if [ "$surface_temp" != "$control_surface_temp" ]; then
        output_dir="RCE_T${surface_temp}_U${control_basal_velocity}_B${control_beta_coefficient}_M${control_microphysic_mode}"
        setup_file="${setup_dir}/var_surface_temp_${output_dir}_set_up.txt"
        cp "$control_setup_file" "$setup_file"
        sed -i "2s/.*/${output_dir}/" "$setup_file"
        sed -i "3s/.*/${surface_temp}/" "$setup_file"
        ./main_script.sh "$setup_file"
    fi
done