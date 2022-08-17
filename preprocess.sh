#! /bin/bash

DATASET_PATH=/scratch/gpfs/blou/Colgate/TestClose5

module purge
module load anaconda3/2021.5
conda activate nerf

singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.camera_model=OPENCV \
   --ImageReader.single_camera=false \
   --ImageReader.single_camera_per_folder=true


singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

singularity exec --nv /scratch/gpfs/$USER/tf_colmap_latest.sif python /scratch/gpfs/$USER/LLFF/imgs2poses.py $DATASET_PATH