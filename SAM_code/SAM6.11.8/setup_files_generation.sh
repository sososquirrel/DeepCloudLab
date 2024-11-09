#!/bin/bash

# Paths to the main script and the base setup file
base_setup_file="set_up.txt"

# Define the control parameters
control_surface_temp="300"
control_basal_velocity="0"
control_beta_coefficient="1"
control_microphysic_mode="1"

# Define the ranges for each parameter
surface_temp_variations=("295" "298" "302" "305")
basal_velocity_variations=("2.5" "5" "10" "20")
beta_coefficient_variations=("0.01" "0.1" "0.5")
microphysic_modes=("2")


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

# Run variations by changing one parameter at a time

# 1. Varying surface temperature
for surface_temp in "${surface_temp_variations[@]}"; do
    if [ "$surface_temp" != "$control_surface_temp" ]; then
        output_dir="RCE_T${surface_temp}_U${control_basal_velocity}_B${control_beta_coefficient}_M${control_microphysic_mode}"
        setup_file="${setup_dir}/var_surface_temp_${output_dir}_set_up.txt"
        cp "$control_setup_file" "$setup_file"
        sed -i "2s/.*/${output_dir}/" "$setup_file"
        sed -i "3s/.*/${surface_temp}/" "$setup_file"
    fi
done

# 2. Varying basal velocity
for basal_velocity in "${basal_velocity_variations[@]}"; do
    if [ "$basal_velocity" != "$control_basal_velocity" ]; then
        output_dir="RCE_T${control_surface_temp}_U${basal_velocity}_B${control_beta_coefficient}_M${control_microphysic_mode}"
        setup_file="${setup_dir}/var_basal_velocity_${output_dir}_set_up.txt"
        cp "$control_setup_file" "$setup_file"
        sed -i "2s/.*/${output_dir}/" "$setup_file"
        sed -i "4s/.*/${basal_velocity}/" "$setup_file"
    fi
done

# 3. Varying beta coefficient
for beta_coefficient in "${beta_coefficient_variations[@]}"; do
    if [ "$beta_coefficient" != "$control_beta_coefficient" ]; then
        output_dir="RCE_T${control_surface_temp}_U${control_basal_velocity}_B${beta_coefficient}_M${control_microphysic_mode}"
        setup_file="${setup_dir}/var_beta_coefficient_${output_dir}_set_up.txt"
        cp "$control_setup_file" "$setup_file"
        sed -i "2s/.*/${output_dir}/" "$setup_file"
        sed -i "5s/.*/${beta_coefficient}/" "$setup_file"
    fi
done

# 4. Varying microphysics mode
for microphysic_mode in "${microphysic_modes[@]}"; do
    if [ "$microphysic_mode" != "$control_microphysic_mode" ]; then
        output_dir="RCE_T${control_surface_temp}_U${control_basal_velocity}_B${control_beta_coefficient}_M${microphysic_mode}"
        setup_file="${setup_dir}/var_microphysic_mode_${output_dir}_set_up.txt"
        cp "$control_setup_file" "$setup_file"
        sed -i "2s/.*/${output_dir}/" "$setup_file"
        sed -i "6s/.*/${microphysic_mode}/" "$setup_file"
    fi
done

