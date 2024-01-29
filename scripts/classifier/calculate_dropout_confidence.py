# this script evaluates an existing model with MC dropout
# this is needed for confidence estimation

import argparse
import meld_graph
import meld_graph.models
import meld_graph.experiment
import meld_graph.dataset
import meld_graph.data_preprocessing
import meld_graph.evaluation

from meld_graph.dataset import GraphDataset
from meld_classifier.meld_cohort import MeldCohort

from meld_graph.evaluation import Evaluator
import json
import os
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to evaluate one model on either train, test, val, or trainval. Val as default does not save predictions"""
    )
    parser.add_argument("--model_path", help="path to trained model config")
    parser.add_argument("--split", help="train, test, val, or trainval")
    parser.add_argument("--new_data", help="json file containing new data parameters", default=None)
    parser.add_argument("--model_name", default="ensemble_best_model.pt", help="name of the model to load")
    parser.add_argument('--p', default=0.8, type=float, help='probability of keeping a value. p=1 means no dropout, p=0 means all values are dropped out')
    parser.add_argument('--n', default=10, type=int, help='number of times MC dropout estimation is run')
    args = parser.parse_args()
    exp = meld_graph.experiment.Experiment.from_folder(args.model_path)

    if args.new_data != None:
        args.split = 'test'
        new_data_params = json.load(open(args.new_data))
        exp.data_parameters["hdf5_file_root"] = new_data_params["hdf5_file_root"]
        exp.data_parameters["dataset"] = new_data_params["dataset"]
        cohort = MeldCohort(
                hdf5_file_root=new_data_params["hdf5_file_root"],
                dataset=new_data_params["dataset"],
            )
        subjects, _, _ = cohort.read_subject_ids_from_dataset()
    else:
        if args.split == "trainval":
            subjects = exp.data_parameters["train_ids"] + exp.data_parameters["val_ids"]
        else:
            sub_split = args.split + "_ids"
            subjects = exp.data_parameters[sub_split]
        exp.data_parameters["augment_data"] = {}
        features = exp.data_parameters["features"]
        cohort = MeldCohort(
                hdf5_file_root=exp.data_parameters["hdf5_file_root"],
                dataset=exp.data_parameters["dataset"],
            )
    
    dataset = GraphDataset(subjects, cohort, exp.data_parameters, mode="test")

    if args.new_data != None:
        save_dir = new_data_params['save_dir']
    else:
        save_dir = args.model_path
    
    eva = Evaluator(
        experiment=exp,
        checkpoint_path=args.model_path,
        save_dir=save_dir,
        make_images=False,
        dataset=dataset,
        cohort=cohort,
        subject_ids=subjects,
        mode="test",
        saliency=False,
        model_name=args.model_name,
    )
   
    eva.enable_mc_dropout(p=args.p, n=args.n)

    # set up suffix for saved hdf5 file
    if args.split == "test":
        suffix = ""
    elif args.split == "val":
        suffix = ""
    elif args.split == "train":
        suffix = "_train"
    elif args.split == "trainval":
        suffix = "_trainval"
    else:
        raise NotImplementedError(args.split)
    
    # predict and cluster data
    eva.load_predict_data(save_prediction=True, save_prediction_suffix=suffix)
    eva.threshold_and_cluster(save_prediction_suffix=suffix)

    # save dropout parameters in hdf5
    filename = os.path.join(eva.save_dir, "results", f"predictions{suffix}{eva.dropout_suffix}.hdf5")
    with h5py.File(filename, mode='r+') as f:
        f.attrs['dropout_p'] = args.p
        f.attrs['dropout_n'] = args.n