import os
import numpy as np
import nibabel as nb
import shutil
import scipy
from scipy import stats as st
from os.path import join as opj
import subprocess
from subprocess import Popen
import meld_graph.mesh_tools as mt
from meld_graph.tools_commands_prints import get_m


def get_adj_mat(surf):

        all_edges = np.vstack(
            [surf["faces"][:, :2], surf["faces"][:, 1:3], surf["faces"][:, [2, 0]]]
        )
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones(len(all_edges), np.uint8), (all_edges[:, 0], all_edges[:, 1])),
            shape=(len(surf["coords"]), len(surf["coords"])),
        ).tocsr()
        return adj_mat
    
def cluster(mask,adj_mat):
        """cluster predictions and threshold based on min_area_threshold

        Args:
            mask: boolean mask of the per-vertex lesion predictions to cluster"""
        n_comp, labels = scipy.sparse.csgraph.connected_components(adj_mat[mask][:, mask])
        islands = np.zeros(len(mask))
        island_count=0
        # only include islands larger than minimum size.
        for island_index in np.arange(n_comp):
            include_vec = labels == island_index
            island_count += 1
            island_mask = mask.copy()
            island_mask[mask] = include_vec
            islands[island_mask] = island_count
        return islands
    
def reindex(clustered,prediction):
    """replace cluster numbers"""
    reindexed_clustered = np.zeros(len(clustered),dtype=int)
    for c in np.unique(clustered)[1:]:
        reindexed_clustered[clustered==c]=st.mode(prediction[clustered==c])[0][0]
    return reindexed_clustered

def correct_interpolation_error(input_mgh,input_surf,output_mgh):
    """correct errors due to interpolation from xhemi back to native"""
    prediction = np.array(mt.load_mgh(input_mgh))
    surf = mt.load_mesh_geometry(input_surf)
    adj = get_adj_mat(surf)
    clustered=cluster(prediction>0,adj)
    reindexed_clustered = reindex(clustered,prediction)
    mt.save_mgh(output_mgh,reindexed_clustered,nb.load(input_mgh))
    return

def correct_interpolation_error_v2(input_mgh,input_surf,output_mgh):
    """correct errors due to interpolation from xhemi back to native"""
    prediction = np.array(mt.load_mgh(input_mgh))
    tolerance = 1e-5
    close_to_int = np.isclose(prediction, np.round(prediction),
     rtol=tolerance, atol=tolerance)
    need_assignment = np.where(~close_to_int)[0]
    #for each cluster find nearby nonzero vertices to assign to them.
    #either geodesic distance or neighbours. remove from remainder to be assigned.
    surf = mt.load_mesh_geometry(input_surf)
    neighbours = mt.get_neighbours_from_tris(surf['faces'])
    new_prediction = np.zeros_like(prediction)
    new_prediction[close_to_int] = np.round(prediction[close_to_int])
    while len(need_assignment)>0:
        need_assignment=[]
        for vertex in need_assignment:
            vertex_neighbours = neighbours[vertex]
            vals = new_prediction[vertex_neighbours]
            if (vals>0).any():
                new_prediction = np.max(vals)
            else:
                need_assignment.append(vertex)
    mt.save_mgh(output_mgh,new_prediction,nb.load(input_mgh))
    return

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
    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/lh.prediction.mgh --trg {subjects_dir}/{subject_id}/surf/lh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/{subject_id}/surf/lh.sphere.reg'
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    # Moves the right hemi back from fsaverage to native. There are 2 steps
    #Step1: move left hemi fsaverage to right hemi of fsaverage
    # --src is the source image i.e. the map you want to move back so change to the name of the cluster map in fsaverage_sym that you want to move back. should be rh....
    # --trg is the target image i.e. the name of the map on the rh -  i called these rh.{name of file}_on_rh.mgh - these are still in template space (fsaverage)
    # Step 2: move from rh of fsaverage to native space
    # --src is the source image i.e. the name of the file you created in step1
    # --trg is the target image i.e. the name of the map you want to create in the subject's native space

    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction.mgh --trg {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction_on_rh.mgh --streg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/rh.sphere.left_right'
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    command = f'SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/xhemi/classifier/rh.prediction_on_rh.mgh --trg {subjects_dir}/{subject_id}/surf/rh.prediction.mgh --streg {subjects_dir}/fsaverage_sym/surf/rh.sphere.reg {subjects_dir}/{subject_id}/surf/rh.sphere.reg'
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode!=0:
        print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
        return False

    #correct from interpolation error
    for hemi in ['lh','rh']:

        #correct prediction from float values in interpolation 
        input_file=opj(subjects_dir,subject_id,'surf',f'{hemi}.prediction.mgh')
        output_mgh=opj(subjects_dir,subject_id,'surf',f'{hemi}.prediction_corr.mgh')
        input_surf=opj(subjects_dir,subject_id,'surf',f'{hemi}.white')
        correct_interpolation_error_v2(input_file,input_surf,output_mgh)

        #map from surface back to vol
        command = f'SUBJECTS_DIR={subjects_dir} mri_surf2vol --identity {subject_id} --template {subjects_dir}/{subject_id}/mri/T1.mgz --o {subjects_dir}/{subject_id}/mri/{hemi}.prediction.mgz --hemi {hemi} --surfval {subjects_dir}/{subject_id}/surf/{hemi}.prediction_corr.mgh --fillribbon'
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




