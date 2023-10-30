#! /bin/bash

#SBATCH -o confidence_%j.out
#SBATCH -e confidence_%j.out
#SBATCH -J jupyterlab
#SBATCH -A CORE-WCHN-MELD-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=48000
#SBATCH -t 01:00:00

#! partitions: ampere, pascal
# set up
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load miniconda/3
# load cuda
module load cuda/11.1


source ~/activate_env.sh
conda activate meld_test

# run script
BASE=/home/co-spit1/software/
MODEL_PATH='/rds/project/kw350/rds-kw350-meld/experiments_graph/co-spit1/23-08-30_IGKW_object_save_final/s_0/fold_all'
python $BASE/meld_classifier_GDL/scripts/calculate_dropout_confidence.py --model_path $MODEL_PATH --split test --model_name best_model.pt --p $1 --n 10
