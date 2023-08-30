### called by evaluate.sh which is launched by cross_val_aucs.py
### Runs one model across a val /test cohort and will either save predictions
### or calculate summary statistics
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
import numpy as np
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to evaluate one model on either train, test, val, or trainval. Val as default does not save predictions"""
    )
    parser.add_argument("--model_path", help="path to trained model config")
    parser.add_argument("--split", help="train, test, val, or trainval")
    parser.add_argument("--saliency", action='store_true', default=False, help="calculate integrated gradients saliency")
    parser.add_argument("--new_data", help="json file containing new data parameters", default=None)
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
        saliency=args.saliency,
    )
   
    # only save predictions on test, no need on vals but instead calculate ROCs
    if args.split == "test":
        save_prediction = True
        roc_curves_thresholds = None
        suffix = ""
    elif args.split == "val":
        save_prediction = False
        roc_curves_thresholds = np.linspace(0, 1, 21)
        suffix = ""
    elif args.split == "train":
        save_prediction = True
        roc_curves_thresholds = None
        suffix = "_train"
    elif args.split == "trainval":
        save_prediction = True
        roc_curves_thresholds = None
        suffix = "_trainval"
    else:
        raise NotImplementedError(args.split)

    eva.load_predict_data(
        save_prediction=save_prediction,
        roc_curves_thresholds=roc_curves_thresholds,
        save_prediction_suffix=suffix,
    )

    # # calculate stats
    eva.stat_subjects()
