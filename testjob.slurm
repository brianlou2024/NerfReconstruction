#!/bin/bash
#SBATCH --job-name=test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1      # number of tasks per nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=blou@princeton.edu

DATASET_PATH=/scratch/gpfs/blou/Colgate/TestTemp
EXPERIMENT_NAME=colgatetest
ITERATIONS=10001

module purge
module load anaconda3/2021.5
conda activate nerf

singularity exec --nv /scratch/gpfs/$USER/colmap_latest.sif colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.camera_model=SIMPLE_RADIAL \
   # --ImageReader.camera_model=OPENCV \
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

python run_nerf.py --config colgate_config.txt --datadir $DATASET_PATH --iterations $ITERATIONS --i_weights 1000

python gen_video.py --expname $EXPERIMENT_NAME --iterations $ITERATIONS