### script to optimise the sigmoid parameters on the trainval dataset

import os
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to evaluate one model on either train, test, val, or trainval. Val as default does not save predictions"""
    )
    parser.add_argument("--model_path", help="path to trained model config")
    parser.add_argument("--model_name", default="ensemble_best_model.pt", help="name of the model to load")
    args = parser.parse_args()
    exp = meld_graph.experiment.Experiment.from_folder(args.model_path)
    
    #initialise trainval dataset
    subjects = exp.data_parameters["train_ids"] + exp.data_parameters["val_ids"]
    exp.data_parameters["augment_data"] = {}
    features = exp.data_parameters["features"]
    cohort = MeldCohort(
            hdf5_file_root=exp.data_parameters["hdf5_file_root"],
            dataset=exp.data_parameters["dataset"],
        )
    dataset = GraphDataset(subjects, cohort, exp.data_parameters, mode="test")

    #save results in the model_path 
    save_dir = args.model_path
   
    # set thresholding and clustering to false
    thresh_and_clust = False

    # create evaluator
    eva = Evaluator(
        experiment=exp,
        checkpoint_path=args.model_path,
        save_dir=save_dir,
        make_images=False,
        dataset=dataset,
        cohort=cohort,
        subject_ids=subjects,
        mode="test",
        thresh_and_clust=thresh_and_clust,
        model_name=args.model_name,
    )
   
    save_prediction = True
    roc_curves_thresholds = None
    suffix = args.model_name.split('.')[0]
    
    #TODO:need to enable loading pre-existing predictions
    #predict subjects
    eva.load_predict_data(
        save_prediction=save_prediction,
        roc_curves_thresholds=roc_curves_thresholds,
        save_prediction_suffix=suffix,
    )

    # optimise sigmoid based on predictions
    eva.optimise_sigmoid(ymin_r=[0.01,0.03,0.05], ymax_r=[0.3,0.4,0.5], k_r=[1], m_r=[0.1,0.05], suffix=suffix) 

