## This script runs a FreeSurfer reconstruction on a participant
## Within your  MELD folder should be an input folder that contains folders 
## for each participant. Within each participant folder should be a T1 folder 
## that contains the T1 in nifti format ".nii" and where available a FLAIR 
## folder that contains the FLAIR in nifti format ".nii"

## To run : python run_script_segmentation.py -id <sub_id> -harmo_code <harmo_code>


import os
from tabnanny import verbose
import numpy as np
from sqlite3 import paramstyle
import sys
import argparse
import subprocess
from subprocess import Popen
# from subprocess import Popen, DEVNULL, STDOUT, check_call
import threading
import multiprocessing
from functools import partial
import tempfile
import glob
import shutil
import pandas as pd
from os.path import join as opj
from meld_graph.paths import BASE_PATH, MELD_DATA_PATH, DEMOGRAPHIC_FEATURES_FILE, FS_SUBJECTS_PATH, CLIPPING_PARAMS_FILE
from meld_graph.tools_pipeline import get_m, create_demographic_file, get_anat_files
from scripts.data_preparation.extract_features.create_xhemi import run_parallel_xhemi, create_xhemi
from scripts.data_preparation.extract_features.create_training_data_hdf5 import create_training_data_hdf5
from scripts.data_preparation.extract_features.sample_FLAIR_smooth_features import sample_flair_smooth_features
from scripts.data_preparation.extract_features.lesion_labels import lesion_labels
from scripts.data_preparation.extract_features.move_to_xhemi_flip import move_to_xhemi_flip


def init(lock):
    global starting
    starting = lock

def check_FS_outputs(folder):
    FS_complete=True
    surf_files = ['pial','white','sphere']
    hemis=['lh','rh']
    for sf in surf_files:
        for hemi in hemis:
            fname = opj(folder,'surf',f'{hemi}.{sf}')
            if not os.path.isfile(fname):
                return False
    return FS_complete

def check_xhemi_outputs():
    #TODO
    pass

def fastsurfer_subject(subject, fs_folder, verbose=False):
    # run fastsurfer segmentation on 1 subject
    subject_id = subject['id']
    subject_t1_path = subject['T1_path']
    
    # get subject folder
    # if freesurfer outputs already exist for this subject, continue running from where it stopped
    # else, run FS
    if os.path.isdir(opj(fs_folder, subject_id)):
        if check_FS_outputs(opj(fs_folder, subject_id))==True:
            print(get_m(f'Fastsurfer outputs already exists for subject {subject_id}. Freesurfer will be skipped', subject_id, 'STEP 1'))
            return True
        if check_FS_outputs(opj(fs_folder, subject_id))==False:
            print(get_m(f'Fastsurfer outputs already exists for subject {subject_id} but is incomplete. Delete folder {opj(fs_folder, subject_id)} and reran', subject_id, 'ERROR'))
            return False
    else:
        pass  
    
    # for parallelisation
    starting.acquire()  # no other process can get it until it is released

    # setup cortical segmentation command
    print(get_m(f'Segmentation using T1 only with FastSurfer', subject_id, 'INFO'))
    python_version = 'python'+'.'.join(sys.version.split('.')[0:2])
    print(python_version)
    command = format(
        "$FASTSURFER_HOME/run_fastsurfer.sh --sd {} --sid {} --t1 {} --parallel --batch 1 --fsaparc --py {}".format(fs_folder, subject_id, subject_t1_path, python_version)
    )

    # call fastsurfer
    from subprocess import Popen
    print(get_m('Start cortical parcellation (up to 2h). Please wait', subject_id, 'INFO'))
    print(get_m(f'Results will be stored in {fs_folder}/{subject_id}', subject_id, 'INFO'))
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    threading.Timer(120, starting.release).start()  # release in two minutes
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode==0:
        print(get_m(f'Finished cortical parcellation', subject_id, 'INFO'))
        return True
    else:
        print(get_m(f'Cortical parcellation using fastsurfer failed. Please check the log at {fs_folder}/{subject_id}/scripts/recon-all.log', subject_id, 'ERROR'))
        print(get_m(f'COMMAND failing : {command} with error {stderr}', None, 'ERROR'))
        return False


def fastsurfer_flair(subject, fs_folder, verbose=False):
    #improve fastsurfer segmentation with FLAIR on 1 subject

    subject_id = subject['id']
    subject_flair_path = subject['FLAIR_path']

    if os.path.isfile(opj(fs_folder, subject_id, "mri", "FLAIR.mgz")):
        print(get_m(f'Freesurfer FLAIR reconstruction outputs already exists. FLAIRpial will be skipped', subject_id, 'STEP 1'))
        return

    if subject_flair_path == None:
        return 

    print(get_m("Starting FLAIRpial", subject_id, 'INFO'))
    command = format(
        "recon-all -sd {} -subject {} -FLAIR {} -FLAIRpial -autorecon3".format(fs_folder, subject_id, subject_flair_path)
    )
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode==0:
        print(get_m(f'Finished FLAIRpial reconstruction', subject_id, 'INFO'))
        return True
    else:
        print(get_m(f'FLAIRpial reconstruction failed. Please check the log at {fs_folder}/{subject_id}/scripts/recon-all.log', subject_id, 'ERROR'))
        print(get_m(f'COMMAND failing : {command} with error {stderr}', None, 'ERROR'))
        return False

def freesurfer_subject(subject, fs_folder, verbose=False):
    #run freesurfer recon-all segmentation on 1 subject

    subject_id = subject['id']
    subject_t1_path = subject['T1_path']
    subject_flair_path = subject['FLAIR_path']

    # get subject folder
    # If freesurfer outputs already exist for this subject, continue running from where it stopped
    # Else, run FS
    if os.path.isdir(opj(fs_folder, subject_id)):
        if check_FS_outputs(opj(fs_folder, subject_id))==True:
            print(get_m(f'Freesurfer outputs already exists for subject {subject_id}. Freesurfer will be skipped', subject_id, 'STEP 1'))
            return True
        if check_FS_outputs(opj(fs_folder, subject_id))==False:
            print(get_m(f'Freesurfer outputs already exists for subject {subject_id} but is incomplete. Delete folder {opj(fs_folder, subject_id)} and reran', subject_id, 'ERROR'))
            return False
    else:
        pass 

    # setup cortical segmentation command
    if subject_flair_path != None:
        print(get_m('Segmentation using T1 and FLAIR with Freesurfer', subject_id, 'STEP 1'))
        command = format(
            "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -FLAIR {} -FLAIRpial -all".format(
                fs_folder, subject_id, subject_t1_path, subject_flair_path
            )
        )
    else:
        print(get_m('Segmentation using T1 only with Freesurfer', subject_id, 'STEP 1'))
        command = format(
            "$FREESURFER_HOME/bin/recon-all -sd {} -s {} -i {} -all".format(fs_folder, subject_id, subject_t1_path)
        )

    # call Freesurfer
    print(get_m('Start cortical parcellation (up to 6h). Please wait', subject_id, 'INFO'))
    print(get_m(f'Results will be stored in {fs_folder}', subject_id, 'INFO'))
    starting.acquire()  # no other process can get it until it is released
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    threading.Timer(120, starting.release).start()  # release in two minutes
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode==0:
        print(get_m(f'Finished cortical parcellation', subject_id, 'INFO'))
        return True
    else:
        print(get_m(f'Cortical parcellation using Freesurfer failed. Please check the log at {fs_folder}/{subject_id}/scripts/recon-all.log', subject_id, 'ERROR'))
        print(get_m(f'COMMAND failing : {command} with error {stderr}', None, 'ERROR'))
        return False
    

def extract_features(subject_id, fs_folder, output_dir, verbose=False):
    
    #check FS outputs
    if check_FS_outputs(opj(fs_folder, subject_id))==False:
        print(get_m(f'Files are missing in Freesurfer outputs for subject {subject_id}. Check {opj(fs_folder, subject_id)} is complete before re-running', subject_id, 'ERROR'))
        return False
    else:
        pass 

    # Launch script to extract surface-based features from freesurfer outputs
    print(get_m('Extract surface-based features', subject_id, 'STEP 2'))
    
    #### EXTRACT SURFACE-BASED FEATURES #####
    # Create the output directory to store the surface-based features processed
    os.makedirs(output_dir, exist_ok=True)
    
    #register to symmetric fsaverage xhemi
    print(get_m(f'Creating registration to template surface', subject_id, 'INFO'))
    result = create_xhemi(subject_id, fs_folder, verbose=verbose)
    if result == False:
        return False

    #create basic features
    print(get_m(f'Sampling features in native space', subject_id, 'INFO'))
    result = sample_flair_smooth_features(subject_id, fs_folder, verbose=verbose)
    if result == False:
        return False

    #move features and lesions to template
    print(get_m(f'Moving features to template surface', subject_id, 'INFO'))
    result = move_to_xhemi_flip(subject_id, fs_folder, verbose = verbose )
    if result == False:
        return False

    print(get_m(f'Moving lesion masks to template surface', subject_id, 'INFO'))
    result = lesion_labels(subject_id, fs_folder, verbose=verbose)
    if result == False:
        return False

    #create training_data matrix for all patients and controls.
    print(get_m(f'Creating final training data matrix', subject_id, 'INFO'))
    result = create_training_data_hdf5(subject_id, fs_folder, output_dir  )
    if result == False:
        return False
 
def run_subjects_segmentation_parallel(subject_ids, num_procs=10, harmo_code="noHarmo", use_fastsurfer=False, verbose=False):
    # parallel version of the pipeline, finish each stage for all subjects first

    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    proc = Popen(ini_freesurfer, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    proc.communicate()
    if proc.returncode!=0:
        print(get_m(f'Could not initialise Freesurfer. Check that it is installed and that $FREESURFER_HOME exists', None, 'ERROR'))
        return False

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    ## create dictionary with T1 and FLAIR paths
    subjects_dict = np.array([get_anat_files(subject_id) for subject_id in subject_ids])
    
    if use_fastsurfer:
        ## first processing stage with fastsurfer: segmentation
        pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
        subject_ids_failed=[]
        mask=[]
        for i,result in enumerate(pool.imap(partial(fastsurfer_subject, fs_folder=fs_folder, verbose=verbose), subjects_dict)):
            if result==False:
                print(get_m(f'Subject removed from futur process because a step in the pipeline failed', subject_ids[i], 'ERROR'))
                subject_ids_failed.append(subject_ids[i])
                mask.append(False)
            else:
                mask.append(True)
                pass
        #update list subjects 
        subject_ids = list(set(subject_ids).difference(subject_ids_failed))
        subjects_dict = subjects_dict[mask]
        
        ## flair pial correction
        pool = multiprocessing.Pool(processes=num_procs)
        subject_ids_failed=[]
        mask=[]
        for i,result in enumerate(pool.imap(partial(fastsurfer_flair, fs_folder=fs_folder, verbose=verbose), subjects_dict)):
            if result==False:
                print(get_m(f'Subject removed from futur process because a step in the pipeline failed', subject_ids[i], 'ERROR'))
                subject_ids_failed.append(subject_ids[i])
                mask.append(False)
            else:
                mask.append(True)
                pass    
        subject_ids = list(set(subject_ids).difference(subject_ids_failed))
        subjects_dict = subjects_dict[mask]
    else:
        ## processing with freesurfer: segmentation
        pool = multiprocessing.Pool(processes=num_procs, initializer=init, initargs=[multiprocessing.Lock()])
        subject_ids_failed=[]
        for i,result in enumerate(pool.imap(partial(freesurfer_subject, fs_folder=fs_folder, verbose=verbose), subjects_dict)):
            if result==False:
                print(get_m(f'Subject removed from futur process because a step in the pipeline failed', subject_ids[i], 'ERROR'))
                subject_ids_failed.append(subject_ids[i])
            else:
                pass
        subject_ids = list(set(subject_ids).difference(subject_ids_failed))


    ### EXTRACT SURFACE-BASED FEATURES ###
    print(get_m(f'Extract surface-based features', subject_ids, 'STEP 2'))
    output_dir = opj(BASE_PATH, f"MELD_{harmo_code}")

    # parallelize create xhemi because it takes a while!
    print(get_m(f'Run create xhemi in parallel', subject_ids, 'INFO'))
    subject_ids = run_parallel_xhemi(subject_ids, fs_folder, num_procs=num_procs, verbose=verbose)

    # Launch script to extract features
    subject_ids_failed=[]
    for i,subject in enumerate(subject_ids):
        print(get_m(f'Extract features in hdf5', subject, 'INFO'))
        result = extract_features(subject, fs_folder=fs_folder, output_dir=output_dir, verbose=verbose)
        if result==False:
            print(get_m(f'Subject removed from futur process because a step in the pipeline failed', subject_ids[i], 'ERROR'))
            subject_ids_failed.append(subject_ids[i])
    subject_ids = list(set(subject_ids).difference(subject_ids_failed))

    return subject_ids

def run_subject_segmentation(subject_id, harmo_code="noHarmo", use_fastsurfer=False, verbose=False):
    # pipeline to segment the brain, exract surface-based features for 1 subject
        
    ### SEGMENTATION ###
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    proc = Popen(ini_freesurfer, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    result = proc.communicate()
    if proc.returncode!=0:
        print(get_m(f'Could not initialise Freesurfer. Check that it is installed and that $FREESURFER_HOME exists', None, 'ERROR'))
        return False
        

    ## Make a directory for the outputs
    fs_folder = FS_SUBJECTS_PATH
    os.makedirs(fs_folder, exist_ok=True)

    ## create dictionary with T1 and FLAIR paths
    subject_dict = get_anat_files(subject_id)
    
    if use_fastsurfer:
        ## first processing stage with fastsurfer: segmentation
        init(multiprocessing.Lock())
        result = fastsurfer_subject(subject_dict,fs_folder, verbose=verbose)
        if result == False:
            return False

        ## flair pial correction
        init(multiprocessing.Lock())
        result = fastsurfer_flair(subject_dict,fs_folder, verbose=verbose)
        if result == False:
            return False
    else:
        ## processing with freesurfer: segmentation
        init(multiprocessing.Lock())
        result = freesurfer_subject(subject_dict,fs_folder, verbose=verbose)
        if result == False:
            return False
    
    ### EXTRACT SURFACE-BASED FEATURES ###
    output_dir = opj(BASE_PATH, f"MELD_{harmo_code}")
    result = extract_features(subject_id, fs_folder=fs_folder, output_dir=output_dir, verbose=verbose)
    if result == False:
            return False


def run_script_segmentation(list_ids=None, sub_id=None, harmo_code='noHarmo', use_parallel=False, use_fastsurfer=False, verbose=False ):
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
    
    if subject_id != None:
        #launch segmentation and feature extraction for 1 subject
        result = run_subject_segmentation(subject_id,  harmo_code = harmo_code, use_fastsurfer = use_fastsurfer, verbose=verbose)
        if result == False:
            print(get_m(f'One step of the pipeline has failed. Process has been aborted for this subject', subject_id, 'ERROR'))
            return False
    else:
        if use_parallel:
            #launch segmentation and feature extraction in parallel
            print(get_m(f'Run subjects in parallel', None, 'INFO'))
            subject_ids_succeed = run_subjects_segmentation_parallel(subject_ids, harmo_code = harmo_code, use_fastsurfer = use_fastsurfer, verbose=verbose)
            subject_ids_failed= list(set(subject_ids).difference(subject_ids_succeed))
            if len(subject_ids_failed):
                print(get_m(f'One step of the pipeline has failed. Process has been aborted for subjects {subject_ids_failed}', None, 'ERROR'))
                return False
        else:
            #launch segmentation and feature extraction for each subject one after another
            print(get_m(f'Run subjects one after another', None, 'INFO'))
            subject_ids_failed=[]
            for subj in subject_ids:
                result = run_subject_segmentation(subj,  harmo_code = harmo_code, use_fastsurfer = use_fastsurfer, verbose=verbose)
                if result == False:
                    print(get_m(f'One step of the pipeline has failed. Process has been aborted for this subject', subj, 'ERROR'))
                    subject_ids_failed.append(subj)
            if len(subject_ids_failed)>0:
                print(get_m(f'One step of the pipeline has failed and process has been aborted for subjects {subject_ids_failed}', None, 'ERROR'))
                return False

if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description="perform cortical parcellation using recon-all from freesurfer")
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
    parser.add_argument("--fastsurfer", 
                        help="use fastsurfer instead of freesurfer", 
                        required=False, 
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--parallelise", 
                        help="parallelise segmentation", 
                        required=False,
                        default=False,
                        action="store_true",
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
       
       
    run_script_segmentation(
                        harmo_code = args.harmo_code,
                        list_ids=args.list_ids,
                        sub_id=args.id, 
                        use_parallel=args.parallelise, 
                        use_fastsurfer=args.fastsurfer,
                        verbose = args.debug_mode
                        )
    





















