## This script open freeview with MRI images, MELD predictions and surfaces for quality check of segmentation


## To run : python new_pt_qc_script.py -id <sub_id>


import os
import sys
import argparse
import subprocess as sub
import bids.layout
import json
import glob

def return_meld_T1_FLAIR(meld_dir, subject_id):
    subject_data={}
    subject_data['id'] = subject_id
    for modality in ['T1', 'FLAIR']:
        files = glob.glob(os.path.join(meld_dir, subject_id, modality, "*.nii*"))
        if len(files)==1:
            subject_data[f"{modality}_path"] = files[0]
        elif len(files)>1:
            print((f'Find too much volumes for {modality}. Check and remove the additional volumes with same key name', subject_id, 'WARNING'))
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
            print(f'Find too much volumes for {modality}. Check and remove the additional volumes with same key name', subject_id, 'WARNING')
            return None
        else:
            subject_data[f"{modality}_path"] = None
    return subject_data

def get_anat_files(subject_id, meld_data_path):
    ''' 
    return path of T1 and FLAIR if BIDs format or MELD format
    '''
    input_dir = os.path.join(meld_data_path, "input")
    subject_data_meld = return_meld_T1_FLAIR(input_dir, subject_id)
    if subject_data_meld is None:
        return None
    if subject_data_meld['T1_path'] is None:
        subject_data_bids = return_bids_T1_FLAIR(input_dir, subject_id)
        if subject_data_bids is None:
            return None
        if subject_data_bids['T1_path'] is None:
            print(f'ERROR: Could not find any T1w nifti file. Please ensure your data are in MELD or BIDS format')
            return None
        else:
            subject_data = subject_data_bids
    else:
        subject_data = subject_data_meld
    print(f'INFO: T1 file used : {subject_data[f"T1_path"]} ')
    if subject_data['FLAIR_path'] is None:
        print(f'INFO: No FLAIR found')
    else:
        print(f'ERROR: FLAIR file used : {subject_data[f"FLAIR_path"]}')
    
    return subject_data
            
def return_file(path, file_name):
    files = glob.glob(path)
    if len(files)>1 :
        print(f'ERROR: Find too much volumes for {file_name}. Check and remove the additional volumes with same key name') 
        return None
    elif not files:
        print(f'ERROR: Could not find {file_name} volume. Check if name follow the right nomenclature')
        return None
    else:
        return files[0]

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='perform cortical parcellation using recon-all from freesurfer')
    parser.add_argument('-id','--id_subj',
                        help='Subject ID.',
                        required=True,)
    parser.add_argument('-meld_data','--meld_data',
                        help='MELD data folder.',
                        required=True,)
    args = parser.parse_args()
    subject=str(args.id_subj)
    meld_data_path=args.meld_data
    
    # get subject folder and fs folder 
    subject_dir = os.path.join(meld_data_path,'input', subject)
    pred_dir = os.path.join(meld_data_path,'output', 'predictions_reports', subject)
    subject_fs_folder = os.path.join(meld_data_path, 'output', 'fs_outputs', subject)
    
    #initialise freesurfer variable environment
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
        
    # Find inputs T1 and FLAIR if exists
    if not os.path.isdir(subject_fs_folder):
        print(f'Freesurfer outputs does not exist for this subject. Unable to perform qc')
    else : 
        subject_dict = get_anat_files(subject, meld_data_path)
        #select inputs files T1 and FLAIR
        T1_file = subject_dict['T1_path']
        FLAIR_file = subject_dict['FLAIR_path']
        #select predictions files
        pred_lh_file = return_file(os.path.join(pred_dir, 'predictions', 'lh.prediction.nii*'), 'lh_prediction')
        pred_rh_file = return_file(os.path.join(pred_dir, 'predictions', 'rh.prediction.nii*'), 'rh_prediction')
        
        #setup cortical segmentation command
        file_text = os.path.join(meld_data_path, 'temp1.txt')
        if T1_file:
            #create txt file with freeview commands
            with open(file_text, 'w') as f:
                f.write(f'-v {T1_file}:colormap=grayscale -layout 2 \n')
                if FLAIR_file:
                    f.write(f'-v {FLAIR_file}:colormap=grayscale \n')
                if (pred_lh_file!=None) & (pred_rh_file!=None):
                    f.write(f'-v {pred_lh_file}:colormap=lut \n')
                    f.write(f'-v {pred_rh_file}:colormap=lut \n')
                f.write(f'-f {subject_fs_folder}/surf/lh.white:edgecolor=yellow {subject_fs_folder}/surf/lh.pial:edgecolor=red {subject_fs_folder}/surf/rh.white:edgecolor=yellow {subject_fs_folder}/surf/rh.pial:edgecolor=red \n')
            #launch freeview
            freeview = format(f"freeview -cmd {file_text}")
            command = ini_freesurfer + ';' + freeview
            print(f"INFO : Open freeview")
            sub.check_call(command, shell=True)
            os.remove(file_text)
            
        else:
            print('Could not find either T1 volume')
            pass
    

    
