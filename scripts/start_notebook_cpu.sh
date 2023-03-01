#! /bin/bash

#SBATCH -o start_notebook_%j.out
#SBATCH -e start_notebook_%j.out
#SBATCH -J jupyterlab
#SBATCH -A CORE-WCHN-MELD-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p skylake-himem
#SBATCH --mem=24000
#SBATCH -t 08:00:00

#! partitions: ampere, pascal
# set up
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load miniconda/3

source ~/activate_env.sh
conda activate meld_test

# run script
BASE=/home/co-spit1/software/
# prevent othher people from reading the log file
chmod og-r start_notebook_${SLURM_JOB_ID}.out
jupyter notebook --no-browser --ip=* --port=8081 --notebook-dir $BASE
