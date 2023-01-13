#! /bin/bash

DATASET_PATH=/scratch/gpfs/blou/Colgate/Paper_nerf
EXPERIMENT_NAME=Paper_nerf
ITERATIONS=10000
CONFIG=configs/config_sphere.txt

module purge
module load anaconda3/2021.5
conda activate nerf


preprocess(){
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
    --Mapper.init_min_num_inliers=100 --Mapper.abs_pose_min_num_inliers=10 \
    --Mapper.init_num_trials=300 --Mapper.max_reg_trials=10 --Mapper.init_max_reg_trials=10 \
    --Mapper.multiple_models=true --Mapper.min_num_matches=5 \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

singularity exec --nv /scratch/gpfs/blou/colmap_latest.sif colmap bundle_adjuster \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0 \
    --BundleAdjustment.refine_principal_point 1

singularity exec --nv /scratch/gpfs/blou/colmap_latest.sif colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0 \
    --output_type TXT

python utils.py \
    --operation remove_images \
    --datadir $DATASET_PATH
}

# while [ ! -f "$DATASET_PATH/database.db" ]
# do
#     echo "Database not found, re-running COLMAP"
#     preprocess
# done

# singularity exec --nv /scratch/gpfs/$USER/tf_colmap_latest.sif python /scratch/gpfs/$USER/LLFF/imgs2poses.py $DATASET_PATH

# singularity exec --nv /scratch/gpfs/$USER/tensorflow_22.12-tf1-py3.sif python run_nerf.py --config $CONFIG --expname $EXPERIMENT_NAME --datadir $DATASET_PATH --iterations $ITERATIONS --i_testset=10000

singularity exec --nv /scratch/gpfs/$USER/tensorflow_22.12-tf1-py3.sif python gen_video.py --expname $EXPERIMENT_NAME --iterations $ITERATIONS --datadir $DATASET_PATH