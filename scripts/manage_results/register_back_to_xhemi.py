import os
import shutil
from os.path import join as opj
import subprocess
from subprocess import Popen
from meld_graph.tools_pipeline import get_m


def register_subject_to_xhemi(subject_id, subjects_dir, output_dir, template = 'fsaverage_sym', verbose=False):
    ''' move the predictions from fsaverage to native space
    inputs:
        subject_id :  subject ID 
        subjects_dir :  freesurfer subjects directory 
        output_dir :  directory to save final prediction in native space
    '''
    
    #copy template
    if not os.path.isdir(opj(subjects_dir,template)):
        shutil.copytree(opj(os.environ['FREESURFER_HOME'],'subjects',template), opj(subjects_dir, os.path.basename(template)))
 
    # Moves left hemi from fsaverage to native space
    # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back
    # --trg is the target image i.e. the name of the map you want to create in the subject's native space
    # the rest is the registration files
    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/lh.prediction.mgh --trg {subjects_dir}/{subject_id}/surf/lh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/{subject_id}/surf/lh.sphere.reg --nnf '
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    # Moves the right hemi back from fsaverage to native. There are 2 steps
    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction.mgh --trg {subjects_dir}/{subject_id}/surf/rh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg --nnf'
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    #correct from interpolation error
    for hemi in ['lh','rh']:

        #map from surface back to vol
        command = f'SUBJECTS_DIR={subjects_dir} mri_surf2vol --identity {subject_id} --template {subjects_dir}/{subject_id}/mri/T1.mgz --o {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --hemi {hemi} --surfval {subjects_dir}/{subject_id}/surf/{hemi}.prediction.mgh --fillribbon'
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        #register back to original volume
        command = f'SUBJECTS_DIR={subjects_dir} mri_vol2vol --mov {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --targ {subjects_dir}/{subject_id}/mri/orig/001.mgz  --regheader --o {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --nearest'
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        #convert to nifti
        command = f'SUBJECTS_DIR={subjects_dir} mri_convert {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz {subjects_dir}/{subject_id}/mri/{hemi}.prediction.nii.gz -rt nearest'
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

    #move files
    save_dir=opj(output_dir,subject_id,'predictions')
    os.makedirs(save_dir, exist_ok=True)
        
    shutil.move(f'{subjects_dir}/{subject_id}/mri/lh.prediction.nii.gz', f'{save_dir}/lh.prediction.nii.gz')
    shutil.move(f'{subjects_dir}/{subject_id}/mri/rh.prediction.nii.gz', f'{save_dir}/rh.prediction.nii.gz')
        
    #combine vols from left and right hemis
    command=f'mri_concat --i {save_dir}/lh.prediction.nii.gz --i {save_dir}/rh.prediction.nii.gz --o {save_dir}/prediction.nii.gz --combine'
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False
        
if __name__ == "__main__":
    pass




