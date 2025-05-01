## This script runs the MELD surface-based FCD classifier on the patient using the output features from script 2.
## The predicted clusters are then saved as file " " in the /output/<pat_id>/xhemi/classifier folder
## The predicted clusters are then registered back to native space and saved as a .mgh file in the /output/<pat_id>/classifier folder
## The predicted clusters are then registered back to the nifti volume and saved as nifti in the input/<pat_id>/predictions folder
## Individual reports for each identified cluster are calculated and saved in the input/<pat_id>/predictions/reports folder
## These contain images of the clusters on the surface and on the volumetric MRI as well as saliency reports
## The saliency reports include the z-scored feature values and how "salient" they were to the classifier

## To run : python run_script_prediction.py -ids <text_file_with_ids> -harmo_code <harmo_code>


import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
import argparse
import tempfile
import shutil
from os.path import join as opj
from meld_graph.paths import (FS_SUBJECTS_PATH, 
                              MELD_DATA_PATH,
                              DEMOGRAPHIC_FEATURES_FILE, 
                              DEFAULT_HDF5_FILE_ROOT, 
                              EXPERIMENT_PATH, 
                              MODEL_PATH)
from meld_graph.evaluation import Evaluator
from meld_graph.experiment import Experiment
from meld_graph.meld_cohort import MeldCohort
from scripts.manage_results.register_back_to_xhemi import register_subject_to_xhemi
from scripts.manage_results.move_predictions_to_mgh import move_predictions_to_mgh
from scripts.manage_results.plot_prediction_report import generate_prediction_report
from meld_graph.tools_pipeline import get_m, create_demographic_file, create_dataset_file

import warnings
warnings.filterwarnings("ignore")

def predict_subjects(subject_ids, output_dir, plot_images = False, saliency=False,
    experiment_path=EXPERIMENT_PATH, hdf5_file_root= DEFAULT_HDF5_FILE_ROOT,):       
    ''' function to predict on new subject using trained MELD classifier'''
    
    # create dataset csv
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)

    # load models
    exp = Experiment.from_folder(experiment_path)

    #update experiment 
    exp.cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=tmp.name)
    exp.data_parameters["hdf5_file_root"] = hdf5_file_root
    exp.data_parameters["dataset"] = tmp.name
    exp.data_parameters["augment_data"] = {}
    exp.experiment_path = experiment_path
    
    # launch evaluation
    cohort = MeldCohort(
                hdf5_file_root=exp.data_parameters["hdf5_file_root"],
                dataset=exp.data_parameters["dataset"],
            )
    eva = Evaluator(
        experiment=exp,
        checkpoint_path=experiment_path,
        cohort=cohort,
        subject_ids=subject_ids,
        save_dir=output_dir,
        mode="test",
        model_name="best_model",
        threshold='slope_threshold',
        thresh_and_clust=True,
        saliency=saliency,
        make_images=plot_images,
        
    )
    #predict for the dataset
    eva.load_predict_data(
        save_prediction=True,
        roc_curves_thresholds=None,
        )
    #threshold predictions
    eva.threshold_and_cluster()
    #write results in csv
    eva.stat_subjects()
    #plot images 
    if plot_images: 
        eva.plot_subjects_prediction()
    #compute saliency:
    if saliency:
        eva.calculate_saliency()

def run_script_prediction(list_ids=None, sub_id=None, harmo_code='noHarmo', no_prediction_nifti=False, no_report=False, skip_prediction=False, split=False, verbose=False):
    harmo_code = str(harmo_code)
    subject_id=None
    subject_ids=None
    if list_ids != None:
        list_ids=opj(MELD_DATA_PATH, list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
        except:
            subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
        else:
            sys.exit(get_m(f'Could not open {subject_ids}', None, 'ERROR'))       
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and harmonisation code ...', None, 'ERROR'))
        sys.exit(-1) 
    
    # initialise variables
    model_name = MODEL_PATH
    experiment_path = os.path.join(EXPERIMENT_PATH, model_name)
    subjects_dir = FS_SUBJECTS_PATH
    classifier_output_dir = opj(MELD_DATA_PATH,'output','classifier_outputs', model_name)
    data_dir = opj(MELD_DATA_PATH,'input')
    predictions_output_dir = opj(MELD_DATA_PATH,'output','predictions_reports')
    prediction_file = opj(classifier_output_dir, 'results_best_model', 'predictions.hdf5')
    
    subject_ids_failed=[]
    
    #predict on new subjects
    if not skip_prediction:
        print(get_m(f'Run predictions', subject_ids, 'STEP 1'))
        predict_subjects(subject_ids=subject_ids, 
                        output_dir=classifier_output_dir,  
                        plot_images=True, 
                        saliency=True,
                        experiment_path=experiment_path, 
                        hdf5_file_root= DEFAULT_HDF5_FILE_ROOT)
    else:
        print(get_m(f'Skip predictions', subject_ids, 'STEP 1'))
    if not no_prediction_nifti:        
        #Register predictions to native space
        for i, subject_id in enumerate(subject_ids):
            print(get_m(f'Move predictions into volume', subject_id, 'STEP 2'))
            result = move_predictions_to_mgh(subject_id=subject_id, 
                                subjects_dir=subjects_dir, 
                                prediction_file=prediction_file,
                                verbose=verbose)
            if result == False:
                print(get_m(f'One step of the pipeline has failed. Process has been aborted for this subject', subject_id, 'ERROR'))
                subject_ids_failed.append(subject_id)
                continue
            
            #Register prediction back to nifti volume
            print(get_m(f'Move prediction back to native space', subject_id, 'STEP 3'))
            result = register_subject_to_xhemi(subject_id=subject_id, 
                                        subjects_dir=subjects_dir, 
                                        output_dir=predictions_output_dir, 
                                        verbose=verbose)
            if result == False:
                print(get_m(f'One step of the pipeline has failed. Process has been aborted for this subject', subject_id, 'ERROR'))
                subject_ids_failed.append(subject_id)
                continue
            
        if not no_report:
            # Create individual reports of each identified cluster
            print(get_m(f'Create pdf report', subject_ids, 'STEP 4'))
            generate_prediction_report(
                subject_ids = subject_ids,
                data_dir = data_dir,
                prediction_path=classifier_output_dir,
                experiment_path=experiment_path, 
                output_dir = predictions_output_dir,
                harmo_code = harmo_code,
                hdf5_file_root = DEFAULT_HDF5_FILE_ROOT
            )
        
    if len(subject_ids_failed)>0:
        print(get_m(f'One step of the pipeline has failed and process has been aborted for subjects {subject_ids_failed}', None, 'ERROR'))
        return False

if __name__ == '__main__':
    import scripts.env_setup
    scripts.env_setup.setup()

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'ID' column",
                        required=False,
                        )
    parser.add_argument("-harmo_code","--harmo_code",
                        default="noHarmo",
                        help="Harmonisation code",
                        required=False,
                        )
    parser.add_argument('-demos', '--demographic_file', 
                        type=str, 
                        help='provide the demographic files for the harmonisation',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--no_prediction_nifti',
                        action="store_true",
                        help='Only predict. Does not produce prediction on native T1, nor report',
                        )
    parser.add_argument('--no_report',
                        action="store_true",
                        help='Predict and map back into native T1. Does not produce report',)
    parser.add_argument('--skip_prediction',
                        action="store_true",
                        help='Skip prediction and go straight to registration and report.',)
    parser.add_argument('--split',
                        action="store_true",
                        help='Split subjects list in chunk to avoid data overload',
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    args = parser.parse_args()
    print(args) 
    
    ### Create demographic file for prediction if not provided
    demographic_file_tmp = DEMOGRAPHIC_FEATURES_FILE
    if args.demographic_file is None:
        harmo_code = str(args.harmo_code)
        subject_id=None
        subject_ids=None
        if args.list_ids != None:
            list_ids=os.path.join(MELD_DATA_PATH, args.list_ids)
            try:
                sub_list_df=pd.read_csv(list_ids)
                subject_ids=np.array(sub_list_df.ID.values)
            except:
                subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
            else:
                    sys.exit(get_m(f'Could not open {subject_ids}', None, 'ERROR'))             
        elif args.id != None:
            subject_id=args.id
            subject_ids=np.array([args.id])
        else:
            print(get_m(f'No ids were provided', None, 'ERROR'))
            print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
            sys.exit(-1) 
        create_demographic_file(subject_ids, demographic_file_tmp, harmo_code=harmo_code)
    else:
        shutil.copy(os.path.join(MELD_DATA_PATH,args.demographic_file), demographic_file_tmp)
       

    run_script_prediction(
                        harmo_code = args.harmo_code,
                        list_ids=args.list_ids,
                        sub_id=args.id,
                        no_prediction_nifti = args.no_prediction_nifti,
                        no_report = args.no_report,
                        split = args.split,
                        skip_prediction=args.skip_prediction,
                        verbose = args.debug_mode
                        )
                