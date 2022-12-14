#! /bin/bash

#SBATCH -o train_%j.out
#SBATCH -e train_%j.out
#SBATCH -J train
#SBATCH -A CORE-WCHN-MELD-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
#SBATCH -t 2:00:00

#! partitions: ampere, pascal
# set up
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load miniconda/3
# load cuda
module load cuda/11.1

CONDA_PATH=~/.conda/envs/meld_graph

# run script
BASE=/home/co-ripa1/rds/hpc-work/scripts
echo $1
source activate $CONDA_PATH; $CONDA_PATH/bin/python3 $BASE/meld_classifier_GDL/scripts/run_evaluation_models_valsdata.py
