#!/bin/bash

paths_s3_simu1="s3://sam-simulations/processed_classes/simulation_SAM_RCE_V0_T300_B1_M1"
paths_s3_simu2="s3://sam-simulations/processed_classes/simulation_SAM_RCE_V2.5_T300_B1_M1"
paths_s3_simu3="s3://sam-simulations/processed_classes/simulation_SAM_RCE_V5_T300_B1_M1"
paths_s3_simu4="s3://sam-simulations/processed_classes/simulation_SAM_RCE_V10_T300_B1_M1"
paths_s3_simu5="s3://sam-simulations/processed_classes/simulation_SAM_RCE_V20_T300_B1_M1"


paths_local_simu1="/home/ec2-user/DeepCloudLab/processed_classes/RCE_T300_U0_B1_M1"
paths_local_simu2="/home/ec2-user/DeepCloudLab/processed_classes/RCE_T300_U2.5_B1_M1"
paths_local_simu3="/home/ec2-user/DeepCloudLab/processed_classes/RCE_T300_U5_B1_M1"
paths_local_simu4="/home/ec2-user/DeepCloudLab/processed_classes/RCE_T300_U10_B1_M1"
paths_local_simu5="/home/ec2-user/DeepCloudLab/processed_classes/RCE_T300_U20_B1_M1"


aws s3 sync $paths_s3_simu1 $paths_local_simu1
aws s3 sync $paths_s3_simu2 $paths_local_simu2
aws s3 sync $paths_s3_simu3 $paths_local_simu3
aws s3 sync $paths_s3_simu4 $paths_local_simu4
aws s3 sync $paths_s3_simu5 $paths_local_simu5




