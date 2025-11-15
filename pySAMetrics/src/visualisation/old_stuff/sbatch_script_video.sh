#!/bin/sh

#SBATCH -A glab                  # Replace ACCOUNT with your account name
#SBATCH -N 8
#SBATCH -c 2
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-10:30            # Runtime in D-HH:MM
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL 
#SBATCH --mail-user=sga2133@columbia.edu
#SBATCH --mem=16G # Set the memory limit to 8GB


source ~/miniconda3/etc/profile.d/conda.sh  # Initialize conda
conda activate samenv

#python script_images.py
python script_video.py