#! /bin/bash

DATASET_PATH=/scratch/gpfs/blou/Colgate/Sphere1
EXPERIMENT_NAME=sphere1
ITERATIONS=2000


module purge
module load anaconda3/2021.5
conda activate nerf

python utils.py \
    --operation remove_images \
    --datadir $DATASET_PATH