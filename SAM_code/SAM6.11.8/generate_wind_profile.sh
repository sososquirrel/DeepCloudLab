#!/bin/bash

# Check if the script received a setup file as an argument
if [ -z "$1" ]; then
    echo "Error: No set_up.txt file provided."
    exit 1
fi

setup_file="$1"

# Read values from the set_up.txt file
mapfile -t lines < "$setup_file"

# Assign each line to a specific variable
case_name="${lines[0]}"
output_dir_name="${lines[1]}"
surface_temp="${lines[2]}"
basal_velocity="${lines[3]}"

echo "$casename"
echo "$output_dir_name"
echo "$surface_temp"
echo "$basal_velocity"

# Check if all variables are set
if [ -z "$case_name" ] || [ -z "$output_dir_name" ] || [ -z "$surface_temp" ] || [ -z "$basal_velocity" ]; then
    echo "Error: Not all variables are set in $setup_file"
    exit 1
fi

# Define the profile limits
profile_start=0
profile_mid=15
profile_end=64

# Create a backup of the original snd file
snd_file="${case_name}/snd"
backup_file="${snd_file}.bak"
cp $snd_file $backup_file

# Use awk to update the last column based on the profile
awk -v basal_velocity="$basal_velocity" -v profile_mid="$profile_mid" -v profile_end="$profile_end" '
BEGIN {
    # Set the velocity decrease rate
    decrease_rate = basal_velocity / profile_mid
}
{
    if (NR > 3 && NR < 65) {
        # Determine the vertical profile value
        if (NR <= profile_mid) {
            velocity = basal_velocity - (NR - 4) * decrease_rate
        } else if (NR > profile_mid && NR <= profile_end) {
            velocity = 0
        } else {
            velocity = $5
        }
        # Print the line with updated last column
        print $1, $2, $3, $4, velocity, $6
    } else {
        # Print the header or any line that should not be modified
        print
    }
}' $backup_file > $snd_file
