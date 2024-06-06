import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from meld_classifier.meld_cohort import MeldCohort,MeldSubject
import sklearn.metrics as metrics
from meld_graph.evaluation import load_prediction, sens_spec_curves, roc_curves, plot_roc_multiple
import pandas as pd
import itertools
import seaborn as sns



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

def calculate_roc(model_path, pred_fname,cohort,thresholds):
    # initialize roc dictionary
    roc_dictionary = {"specificity": np.zeros(len(thresholds)),
                      "sensitivity_plus": np.zeros(len(thresholds))}
    #
    save_dir = os.path.join(model_path, 'results_best_model')
    # get list of subjects
    with h5py.File(os.path.join(save_dir, pred_fname), "r") as f:
        subjects = list(f.keys())
    # load individual subject predictions over folds
    for subj in subjects:
            subject_dictionary = load_predictions_for_subject(subj, save_dir, cohort,thresholds,
                                 roc_dictionary, pred_fname=pred_fname)
    auc = calculate_aucs(roc_dictionary)
    return roc_dictionary, auc

def load_predictions_for_subject(subj, save_dir, cohort,thresholds,
                                 roc_dictionary, pred_fname='predictions.hdf'):
    """
    Load and ensemble subject data. Returns subject_dict with keys "input_labels", "borderzone", "result"
    
    Args:
        subj: subject string
        save_dirs: list of models (folds) that should be loaded & ensembled
        cohort: MeldCohort
    """
    s = MeldSubject(subj,cohort=cohort)
    
    # load labels & borderzone
    labels_hemis = {}
    dists = {}
    for hemi in ['lh', 'rh']:
        dists[hemi], labels_hemis[hemi] = s.load_feature_lesion_data(features=['.on_lh.boundary_zone.mgh'], 
                                                                     hemi=hemi, features_to_ignore=[])
        if np.sum(dists[hemi])==0:
                dists[hemi] +=200
    labels = np.hstack([labels_hemis['lh'][cohort.cortex_mask],labels_hemis['rh'][cohort.cortex_mask]])
    borderzones = np.vstack([dists['lh'][cohort.cortex_mask,:],dists['rh'][cohort.cortex_mask,:]]).ravel()<20
    # load predictions
    pred_file = os.path.join(save_dir, pred_fname)
    result_hemis = load_prediction(subj,pred_file, dset='prediction')
    subject_results = np.hstack([result_hemis['lh'],result_hemis['rh']])
    # build results dict
    subject_dictionary={'input_labels':labels,'borderzone':borderzones,'result':subject_results,
                        }
    roc_curves(subject_dictionary, roc_dictionary, thresholds)
    return subject_dictionary
        
def roc_curves(subject_dictionary, roc_dictionary, roc_curves_thresholds):
    """calculate performance at multiple thresholds"""
    for t_i, threshold in enumerate(roc_curves_thresholds):
        predicted = subject_dictionary["result"] >= threshold
        # if we want tpr vs fpr curve too
        # store sensitivity and sensitivity_plus for each patient (has a label)
        if subject_dictionary["input_labels"].sum() > 0:
            roc_dictionary["sensitivity_plus"][t_i] += np.logical_and(predicted, subject_dictionary["borderzone"]).any()
        # store specificity for controls (no label)
        else:
            roc_dictionary["specificity"][t_i] += ~predicted.any()


def calculate_aucs(roc_dictionary):
    import sklearn.metrics as metrics
    x = 1 - roc_dictionary["specificity"] / roc_dictionary["specificity"][-1]
    y2 = roc_dictionary["sensitivity_plus"] / roc_dictionary["sensitivity_plus"][0]
    roc_dictionary['norm_spec'] = roc_dictionary["specificity"] / roc_dictionary["specificity"][-1]
    roc_dictionary['norm_sens'] = y2
    roc_dictionary["auc"] = metrics.auc(x, y2)
    return roc_dictionary["auc"]


def calculate_optimal_threshold(roc_dictionary,thresholds):
    from scipy.signal import find_peaks

    youden_curve = roc_dictionary['norm_spec']+roc_dictionary['norm_sens']
    bigjump = np.argmax(np.diff(youden_curve))
    #find first peak in youden after bigjump

    peaks,_=find_peaks(youden_curve)
    peaks=peaks[peaks>bigjump]
    threshold_best = np.min(peaks)
    return thresholds[threshold_best]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to evaluate one model on either train, test, val, or trainval. Val as default does not save predictions"""
    )
    parser.add_argument("--model_path", help="path to trained model config")
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
    #only need controls now for optimise sigmoid
    pred_fname = 'predictions_trainval.hdf5'
    thresholds = np.linspace(0, 1, 101)
    roc_dictionary_trainval, auc = calculate_roc(args.model_path, pred_fname,cohort,thresholds)
    tb = calculate_optimal_threshold(roc_dictionary_trainval,thresholds)
    df = np.array([tb,0.5]).reshape(-1,1).T
    df = pd.DataFrame(df,columns=['ymin','ymax'])
    df.to_csv(os.path.join(args.model_path,'results_best_model','two_thresholds.csv'))