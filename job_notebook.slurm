#!/bin/bash
#SBATCH --job-name=colgatenotebook   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1      # number of tasks per nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=blou@princeton.edu

module purge
module load anaconda3/2021.5
module load rh/devtoolset/8
module load cudatoolkit/11.3
conda activate nerf

python notebook.py