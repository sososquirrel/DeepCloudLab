#!/bin/bash

dir_path="/burg/home/sga2133/SAM_simulation/SAM6.11.8"


# Check if the setup file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_set_up.txt>"
    exit 1
fi

setup_file="$1"

# Check if the setup file exists
if [ ! -f "$setup_file" ]; then
    echo "Error: File $setup_file does not exist."
    exit 1
fi

# Print the content of the setup file to verify correctness
echo "Content of $setup_file:"
echo "*********printing the file"
cat "$setup_file"

# Read values from the set_up.txt file
mapfile -t lines < "$setup_file"

# Assign each line to a specific variable
case_name="${lines[0]}"
output_dir_name="${lines[1]}"
surface_temp="${lines[2]}"
basal_velocity="${lines[3]}"
beta_coefficient="${lines[4]}"
microphysic_mode="${lines[5]}"


echo "*********reading the file"
echo "$case_name"
echo "$output_dir_name"
echo "$surface_temp"
echo "$basal_velocity"
echo "$beta_coefficient"
echo "$microphysic_mode"

# Check if all variables are set
if [ -z "$case_name" ] || [ -z "$output_dir_name" ] || [ -z "$surface_temp" ] || [ -z "$basal_velocity" ] || [ -z "$microphysic_mode" ]; then
    echo "Error: Not all variables are set in $setup_file"
    exit 1
fi

# Save the case name to CaseName file
echo $case_name > CaseName

# Save the output directory name to OutputDirName file
echo $output_dir_name > OutputDirName

cat OutputDirName

# Determine the microphysics mode string
if [ "$microphysic_mode" -eq 1 ]; then
    microphysic_str="SAM1MOM"
elif [ "$microphysic_mode" -eq 2 ]; then
    microphysic_str="M2005"
else
    echo "Error: Invalid microphysic mode in $setup_file"
    exit 1
fi

# Generate the case_id using the provided surface temperature and basal velocity
case_id="T${surface_temp}_U${basal_velocity}_${microphysic_str}_B${beta_coefficient}_128x128x64"

# Modify the case_id in the CaseName/prm file
prm_file="${dir_path}/${case_name}/prm"
sed -i "3s/.*/caseid = '${case_id}'/" $prm_file

# Modify the surface temperature in the CaseName/prm file
sed -i "31s/.*/tabs_s = ${surface_temp},/" $prm_file

# Call the wind profile generation script with the set_up.txt path
./generate_wind_profile.sh "$setup_file"

sed -i "235s#.*#         lhf = -tau * qstar / ustar / rbot * ${beta_coefficient}#" ${dir_path}/SRC/oceflx.f90

# Modify line 23 in the Build file based on the microphysics mode
if [ "$microphysic_mode" -eq 1 ]; then
    sed -i "23s/.*/setenv MICRO_DIR MICRO_SAM1MOM/" Build
elif [ "$microphysic_mode" -eq 2 ]; then
    sed -i "23s/.*/setenv MICRO_DIR MICRO_M2005/" Build
else
    echo "Error: Invalid microphysic mode in $setup_file"
    exit 1
fi

# Build the project
./Build

# Check if the build was successful
if [ $? -ne 0 ]; then
  echo "Build failed. Exiting."
  exit 1
fi

cd /burg/glab_new/users/sga2133/SAM_simulation_storage/$output_dir_name/WORK

if [ -f "SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM" ]; then
  echo "Build completed successfully, and the executable exists."
else
  echo "Build completed, but the executable SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM was not found. Exiting."
  exit 1
fi

cd /burg/home/sga2133/SAM_simulation/SAM6.11.8/
echo "hello******************************"
cat OutputDirName
# Submit the job

sbatch job_submit.sh

sleep 30
