#!/bin/bash

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
cat "$setup_file"
