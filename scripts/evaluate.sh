#! /bin/bash

#SBATCH -J run-meld
#SBATCH -A CORE-WCHN-MELD-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=00:40:00 
#SBATCH --mail-type=FAIL
#SBATCH -p skylake

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
BASE=/rds/user/co-ripa1/hpc-work/scripts
echo $1
echo $2
source activate $CONDA_PATH; $CONDA_PATH/bin/python3 $BASE/meld_classifier_GDL/scripts/evaluate_single_model.py --model_path $1 --split $2
