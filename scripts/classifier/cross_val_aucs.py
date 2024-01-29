### Script to parallel evaluation on models across folds
### If mode is val it doesn't save predictions, only stats
### if mode is test, it saves individual subject-level predictions in an hdf5
### so that they can then be ensembled using the ensemble.py script.

import argparse
import os
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Function to run across all folds evaluation script in parallel """
    )
    parser.add_argument(
        "--experiment_path",
        help="path to trained model config without fold, i.e. ending s_2",
    )
    parser.add_argument("--split", help="train, val, test, or trainval.")
    args = parser.parse_args()

    for fold in [0,1,2,3,4]: 
        full_path = os.path.join(args.experiment_path,f'fold_0{fold}')
        if os.path.exists(full_path):
            subprocess.call(f"sbatch evaluate_short.sh {full_path} {args.split} best_model", shell=True)
