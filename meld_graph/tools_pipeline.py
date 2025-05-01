import logging as log
import subprocess
import glob
import json
import os
import bids.layout
import pandas as pd
from subprocess import Popen
from meld_graph.paths import MELD_DATA_PATH

def get_m(message, subject=None, type_message='INFO'):
    try:
        if not isinstance(subject, str):
            subject = ' '.join(subject)
        return f'{type_message} - {subject}: {message}'
    except:
        subject = None
        return f'{type_message}: {message}'

def return_meld_T1_FLAIR(meld_dir, subject_id):
    subject_data={}
    subject_data['id'] = subject_id
    for modality in ['T1', 'FLAIR']:
        files = glob.glob(os.path.join(meld_dir, subject_id, modality, "*.nii*"))
        if len(files)==1:
            subject_data[f"{modality}_path"] = files[0]
        elif len(files)>1:
            print(get_m(f'Find too much volumes for {modality}. Check and remove the additional volumes with same key name', subject_id, 'WARNING'))
            return None
        else:
            subject_data[f"{modality}_path"] = None
    return subject_data

def return_bids_T1_FLAIR(bids_dir, subject_id):
    subject_data={}
    subject_data['id'] = subject_id
    if 'sub-' in subject_id:
        subject_id = subject_id.split('sub-')[-1]
    print(subject_id)
    # get bids structure
    layout = bids.layout.BIDSLayout(bids_dir)
    print(layout)
    # find parameters to extract bids file
    config_file = os.path.join(bids_dir, 'meld_bids_config.json')
    with open(config_file, "r") as json_file:
        dict = json.load(json_file)
    # Create query
    for modality in ['T1', 'FLAIR']:
        query = dict[modality]
        query['subject'] = subject_id
        # Get a list of matching files
        files = layout.get(return_type='file', extension=['nii.gz'], **query)
        if len(files)==1:
            subject_data[f"{modality}_path"] = files[0]
        elif len(files)>1:
            print(get_m(f'Find too much volumes for {modality}. Check and remove the additional volumes with same key name', subject_id, 'WARNING'))
            return None
        else:
            subject_data[f"{modality}_path"] = None
    return subject_data

def get_anat_files(subject_id):
    ''' 
    return path of T1 and FLAIR if BIDs format or MELD format
    '''
    input_dir = os.path.join(MELD_DATA_PATH, "input")
    subject_data_meld = return_meld_T1_FLAIR(input_dir, subject_id)
    if subject_data_meld is None:
        return None
    if subject_data_meld['T1_path'] is None:
        subject_data_bids = return_bids_T1_FLAIR(input_dir, subject_id)
        if subject_data_bids is None:
            return None
        if subject_data_bids['T1_path'] is None:
            print(get_m(f'Could not find any T1w nifti file. Please ensure your data are in MELD or BIDS format', subject_id, 'ERROR'))
            return None
        else:
            subject_data = subject_data_bids
    else:
        subject_data = subject_data_meld
    print(get_m(f'T1 file used : {subject_data[f"T1_path"]} ', subject_id, 'INFO'))
    if subject_data['FLAIR_path'] is None:
        print(get_m(f'No FLAIR found', subject_id, 'INFO'))
    else:
        print(get_m(f'FLAIR file used : {subject_data[f"FLAIR_path"]} ', subject_id, 'INFO'))
    
    return subject_data

def run_command(command, verbose=False):
    # if verbose:
    #     print(get_m(command, None, 'COMMAND'))
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8' )
    # if proc.stderr:
    #     raise subprocess.CalledProcessError(
    #             returncode = proc.returncode,
    #             cmd = proc.args,
    #             stderr = proc.stderr
    #             )
    # if (proc.stdout) and (verbose):
    #     print(get_m("Result: {}".format(proc.stdout.decode('utf-8')), None, 'COMMAND'))
    return proc

def create_demographic_file(subjects_ids, save_file, harmo_code='noHarmo'):
    df = pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['ID']=subjects_ids.astype(str)
    df['Harmo code']=[str(harmo_code) for subject in subjects_ids]
    df['Group']=['patient' for subject in subjects_ids]
    df['Scanner']=['XT' for subject in subjects_ids]
    df.to_csv(save_file)
    
def create_dataset_file(subjects_ids, save_file):
    df=pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['subject_id']=subjects_ids
    df['split']=['test' for subject in subjects_ids]
    df.to_csv(save_file)