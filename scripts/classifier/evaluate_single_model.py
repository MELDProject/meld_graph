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
from meld_graph.meld_cohort import MeldCohort

import os
from meld_graph.evaluation import Evaluator
import numpy as np
import json
import os
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to evaluate one model on either train, test, val, or trainval. Val as default does not save predictions"""
    )
    parser.add_argument("--model_path", help="path to trained model config")
    parser.add_argument("--split", help="train, test, val, or trainval")
    parser.add_argument("--saliency", action='store_true', default=False, help="calculate integrated gradients saliency")
    parser.add_argument("--new_data", help="json file containing new data parameters", default=None)
    parser.add_argument("--model_name", default="best_model", help="name of the model to load")
    parser.add_argument("--threshold", default="two_threshold", help="threshold type, can be two_threshold, multi_threshold, max_threshold")
    
    args = parser.parse_args()
    
    # initialise experiment
    exp = meld_graph.experiment.Experiment.from_folder(args.model_path)
    
    # get threshold parameters in function of threshold type
    # if two_threshold, need to run function to get optimised parameters 
    if args.threshold == 'two_threshold':
        threshold_file = os.path.join(exp.experiment_path, 
                                      f'results_{args.model_name}',f'two_thresholds.csv')
        print(threshold_file)
        if not os.path.exists(threshold_file):
            print('Optimised two thresholds parameters not found')
            sys.exit()
    
    thresh_and_clust = True
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
        print(f'New data: {len(subjects)} subjects')
    else:
        if args.split == "trainval":
            subjects = exp.data_parameters["train_ids"] + exp.data_parameters["val_ids"]
        elif args.split == "val":
            sub_split = args.split + "_ids"
            subjects = exp.data_parameters[sub_split]
            thresh_and_clust=False
        else:
            sub_split = args.split + "_ids"
            subjects = exp.data_parameters[sub_split]
        exp.data_parameters["augment_data"] = {}
        features = exp.data_parameters["features"]
        cohort = MeldCohort(
                hdf5_file_root=exp.data_parameters["hdf5_file_root"],
                dataset=exp.data_parameters["dataset"],
            )
    
    # create new directory to save prediction on new data
    if args.new_data != None:
        save_dir = new_data_params['save_dir']
        print(save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.model_path
    dataset = GraphDataset(subjects, cohort, exp.data_parameters, mode="test")

    # launch evaluation
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
        model_name=args.model_name,
        threshold=args.threshold,
        thresh_and_clust=thresh_and_clust,

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

    # threshold and clustering
    
    # calculate stats
    # make images 
    if args.split!='val':
        eva.threshold_and_cluster(save_prediction_suffix=suffix, )
        eva.stat_subjects()    
        eva.plot_subjects_prediction(suffix=suffix,)
        if args.saliency:
            eva.calculate_saliency(save_prediction_suffix=suffix)
