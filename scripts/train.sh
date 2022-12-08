#! /bin/bash

#SBATCH -o train_%j.out
#SBATCH -e train_%j.out
#SBATCH -J train
#SBATCH -A CAMBRC-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
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

source activate ~/.conda/envs/meld_graph

# run script
BASE=/home/kw350/software/gdl
echo $1
~/.conda/envs/meld_graph/bin/python $BASE/meld_classifier_GDL/scripts/train.py --config_file $1
