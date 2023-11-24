#! /bin/bash


#SBATCH -J run-meld
#SBATCH -A CORE-WCHN-MELD-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=02:00:00 
#SBATCH --mail-type=FAIL
#SBATCH -p icelake

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
~/.conda/envs/meld_graph/bin/python $BASE/meld_classifier_GDL/scripts/evaluate_trainval.py --model_path $1 --model_name $2
~/.conda/envs/meld_graph/bin/python $BASE/meld_classifier_GDL/scripts/calculate_thresholds.py --model_path $1 
