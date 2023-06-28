#! /bin/bash

#SBATCH -o train_%j.out
#SBATCH -e train_%j.out
#SBATCH -J train
#SBATCH -A CORE-WCHN-MELD-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=48000
#SBATCH -t 15:00:00

#! partitions: ampere, pascal
# set up
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load miniconda/3
# load cuda
module load cuda/11.1

source activate ~/.conda/envs/meld_graph

# run script
BASE=/rds/user/co-ripa1/hpc-work/scripts
echo $1
~/.conda/envs/meld_graph/bin/python $BASE/meld_classifier_GDL/scripts/train.py --config_file $1
