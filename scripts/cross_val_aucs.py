

import argparse
import os
import subprocess
#import numpy as np

#point to experiment config folder as argument


#loop over folds within experiment folder

#for each fold run evaluation steps on val set.
#to be super quick this should be per fold sbatch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Function to run across all folds evaluation script in parallel """)
    parser.add_argument("--experiment_path", help="path to trained model config without fold, i.e. ending s_2")
    parser.add_argument("--split", help="val or test to run on.")
    args = parser.parse_args()

    for fold in [0,1,2,3,4]: #np.arange(5):
        full_path = os.path.join(args.experiment_path,f'fold_0{fold}')
        subprocess.call(f'sbatch evaluate.sh {full_path} {args.split}',shell=True)