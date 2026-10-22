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
 
    save_dir=opj(output_dir,subject_id,'predictions')
    os.makedirs(save_dir, exist_ok=True)

    # Moves left hemi from fsaverage to native space
    # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back
    # --trg is the target image i.e. the name of the map you want to create in the subject's native space
    # the rest is the registration files
    
    surf_native_dir = opj(save_dir, 'surf_native')
    os.makedirs(surf_native_dir, exist_ok=True)
    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {save_dir}/fsaverage_sym/lh.prediction.mgh --trg {surf_native_dir}/lh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/{subject_id}/surf/lh.sphere.reg --nnf '
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    # Moves the right hemi back from fsaverage to native.
    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {save_dir}/fsaverage_sym/rh.prediction.mgh --trg {surf_native_dir}/rh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg --nnf'
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    vol_freesurfer_dir = opj(save_dir, 'vol_freesurfer')
    vol_native_dir = opj(save_dir, 'vol_native')
    os.makedirs(vol_freesurfer_dir, exist_ok=True)
    os.makedirs(vol_native_dir, exist_ok=True)

    #correct from interpolation error
    for hemi in ['lh','rh']:

        #map from surface back to vol
        command = f'SUBJECTS_DIR={subjects_dir} mri_surf2vol --identity {subject_id} --template {subjects_dir}/{subject_id}/mri/T1.mgz --o {vol_freesurfer_dir}/{hemi}.prediction.mgz --hemi {hemi} --surfval {surf_native_dir}/{hemi}.prediction.mgh --fillribbon'
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        #register back to original volume
        command = f'SUBJECTS_DIR={subjects_dir} mri_vol2vol --mov {vol_freesurfer_dir}/{hemi}.prediction.mgz --targ {subjects_dir}/{subject_id}/mri/orig/001.mgz  --regheader --o {vol_native_dir}/{hemi}.prediction.mgz --nearest'
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        #convert to nifti
        command = f'SUBJECTS_DIR={subjects_dir} mri_convert {vol_native_dir}/{hemi}.prediction.mgz {vol_native_dir}/{hemi}.prediction.nii.gz -rt nearest'
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False


        
    shutil.move(f'{vol_native_dir}/lh.prediction.nii.gz', f'{save_dir}/lh.prediction.nii.gz')
    shutil.move(f'{vol_native_dir}/rh.prediction.nii.gz', f'{save_dir}/rh.prediction.nii.gz')
        
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




