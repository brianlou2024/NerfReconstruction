#! /bin/bash

DATASET_PATH=/scratch/gpfs/blou/Colgate/Sphere2
EXPERIMENT_NAME=sphere2
ITERATIONS=2000

module purge
module load anaconda3/2021.5
conda activate nerf

:start 

singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.camera_model=OPENCV \
   --ImageReader.single_camera=false \
   --ImageReader.single_camera_per_folder=true \
   --SiftExtraction.estimate_affine_shape=true \
   --SiftExtraction.domain_size_pooling=true


singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.guided_matching=true 

mkdir $DATASET_PATH/sparse

singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
    
singularity exec --nv /scratch/gpfs/blou/colmap_latest.sif colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0 \
    --output_type TXT

python utils.py \
    --operation remove_images \
    --datadir $DATASET_PATH

if [ ! -e $DATASET_PATH/database.db ]
then
    echo "Database not found, re-running COLMAP"
    goto start
fi

singularity exec --nv /scratch/gpfs/$USER/tf_colmap_latest.sif python /scratch/gpfs/$USER/LLFF/imgs2poses.py $DATASET_PATH

python run_nerf.py --config config_sphere.txt --expname $EXPERIMENT_NAME --datadir $DATASET_PATH --iterations $ITERATIONS --i_weights 500 

python gen_video.py --expname $EXPERIMENT_NAME --iterations $ITERATIONS --datadir $DATASET_PATH