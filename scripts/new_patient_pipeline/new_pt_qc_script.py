## This script open freeview with MRI images, MELD predictions and surfaces for quality check of segmentation


## To run : python new_pt_qc_script.py -id <sub_id>


import os
import sys
import argparse
import subprocess as sub
import glob
from meld_graph.tools_pipeline import get_anat_files
from meld_graph.paths import MELD_DATA_PATH, FS_SUBJECTS_PATH
            
def return_file(path, file_name):
    files = glob.glob(path)
    if len(files)>1 :
        print(f'Find too much volumes for {file_name}. Check and remove the additional volumes with same key name') 
        return None
    elif not files:
        print(f'Could not find {file_name} volume. Check if name follow the right nomenclature')
        return None
    else:
        return files[0]
    
if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='perform cortical parcellation using recon-all from freesurfer')
    parser.add_argument('-id','--id_subj',
                        help='Subject ID.',
                        required=True,)
    args = parser.parse_args()
    subject=str(args.id_subj)
    
    # get subject folder and fs folder 
    subject_dir = os.path.join(MELD_DATA_PATH,'input', subject)
    pred_dir = os.path.join(MELD_DATA_PATH,'output', 'predictions_reports', subject)
    subject_fs_folder = os.path.join(FS_SUBJECTS_PATH, subject)
    
    #initialise freesurfer variable environment
    ini_freesurfer = format("$FREESURFER_HOME/SetUpFreeSurfer.sh")
        
    # Find inputs T1 and FLAIR if exists
    if not os.path.isdir(subject_fs_folder):
        print(f'Freesurfer outputs does not exist for this subject. Unable to perform qc')
    else : 
        subject_dict = get_anat_files(subject)
        #select inputs files T1 and FLAIR
        T1_file = subject_dict['T1_path']
        FLAIR_file = subject_dict['FLAIR_path']
        #select predictions files
        pred_lh_file = return_file(os.path.join(pred_dir, 'predictions', 'lh.prediction.nii*'), 'lh_prediction')
        pred_rh_file = return_file(os.path.join(pred_dir, 'predictions', 'rh.prediction.nii*'), 'rh_prediction')
        
        #setup cortical segmentation command
        file_text = os.path.join(MELD_DATA_PATH, 'temp1.txt')
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
    

    
