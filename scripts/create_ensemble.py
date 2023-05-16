# create an ensemble model from multiple folds
# adapted from meld_classifier/scripts/classifier/ensemble.py

import json
import numpy as np
import torch
import os
import shutil
import argparse
import logging

from meld_graph.experiment import Experiment
from meld_graph.ensemble import Ensemble

def _update_subj_ids(data_param_file, ensemble_experiments):
    train_ids = []
    for exp in ensemble_experiments:
        params = json.load(open(os.path.join(exp, f"data_parameters_{exp[1]}.json"), "r"))
        train_ids.extend(params["train_ids"])
    train_ids = list(np.unique(train_ids))

    params = json.load(open(data_param_file, "r"))
    params["train_ids"] = train_ids
    params["val_ids"] = []
    json.dump(params, open(data_param_file, "w"), indent=4)


def create_ensemble(experiment_path, ensemble_experiments):
    """
    Creates ensemble model from experiments and stores it in experiment_path/experiment_name
    Sets train_ids to the union of all train_ids in every experiment (fold). val_ids are empty.
    """
    # create new experiment_path and copy over data_parameters and network_parameters
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # use parameters of first experiments for this
    data_param_file = os.path.join(
        ensemble_experiments[0][0], "data_parameters_{}.json".format(ensemble_experiments[0][1])
    )
    shutil.copyfile(data_param_file, os.path.join(experiment_path, "data_parameters_{}.json".format(experiment_name)))
    network_param_file = os.path.join(
        ensemble_experiments[0][0], "network_parameters_{}.json".format(ensemble_experiments[0][1])
    )
    shutil.copyfile(
        network_param_file, os.path.join(experiment_path, "network_parameters_{}.json".format(experiment_name))
    )
    # merge all train ids
    _update_subj_ids(data_param_file, ensemble_experiments)

    # create results dir
    if not os.path.exists(os.path.join(experiment_path, "results")):
        os.makedirs(os.path.join(experiment_path, "results"))

    # create ensemble
    models = []
    for exp_dir in ensemble_experiments:
        exp = Experiment.from_folder(exp_dir)
        exp.load_model(os.path.join(exp.experiment_path, exp.experiment_name, f'fold_0{exp.fold}', 'best_model.pt'))
        models.append(exp.model)
    ensemble = Ensemble(models)
    
    # save ensemble
    torch.save(ensemble.state_dict(), os.path.join(experiment_path, 'ensemble_model.pt'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ensemble trained experiments."
    )
    parser.add_argument(
        "experiment-folder",
        help="Experiments in one folder to ensemble",
    )
    parser.add_argument(
        "--ensemble-experiment-path", default=None, help="experiment path of the resulting ensemble experiment. Default is experiment-folder/fold_all"
    )
    parser.add_argument("--folds", nargs="+", default=range(5), help='folds in experiment-folder that should be ensembled')
    args = parser.parse_args()

    # experiments to be compared
    exp_dirs = [os.path.join(args.experiment_folder, f'fold_{fold:02d}') for fold in args.folds]

    output_dir = args.ensemble_experiment_path
    if output_dir is None:
        output_dir = os.path.join(args.experiment_folder, 'fold_all')

    # create ensemble
    print("Creating ensemble of {}".format(exp_dirs))
    print("Saving ensembled models to {}".format(output_dir))
    create_ensemble(output_dir, exp_dirs)
