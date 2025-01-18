#!/bin/bash

# Paths to download
paths_3d="s3://sam-simulations/outputs/RCE_T300_U0_B1_M1/WORK/NETCDF_files/3D/dataset_3d.nc"
paths_2d="s3://sam-simulations/outputs/RCE_T300_U0_B1_M1/WORK/NETCDF_files/2D/RCE_T300_U0_SAM1MOM_B1_128x128x64_64.2Dcom_1.nc"
paths_1d="s3://sam-simulations/outputs/RCE_T300_U0_B1_M1/WORK/NETCDF_files/1D/RCE_T300_U0_SAM1MOM_B1_128x128x64.nc"

# Local destinations
local_3d="/home/ec2-user/DeepCloudLab/data/RCE_T300_U0_B1_M1/NETCDF_files/3D/dataset_3d.nc"
local_2d="/home/ec2-user/DeepCloudLab/data/RCE_T300_U0_B1_M1/NETCDF_files/2D/RCE_T300_U0_SAM1MOM_B1_128x128x64_64.2Dcom_1.nc"
local_1d="/home/ec2-user/DeepCloudLab/data/RCE_T300_U0_B1_M1/NETCDF_files/1D/RCE_T300_U0_SAM1MOM_B1_128x128x64.nc"

# Download files
aws s3 cp $paths_3d $local_3d
aws s3 cp $paths_2d $local_2d
aws s3 cp $paths_1d $local_1d
