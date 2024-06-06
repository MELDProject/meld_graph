import os
from os.path import join as opj
import subprocess
from subprocess import Popen
import shutil
import argparse
import numpy as np
import nibabel as nb
from scripts.data_preparation.extract_features import io_meld
from scripts.data_preparation.extract_features.create_identity_reg import create_identity
from meld_graph.tools_pipeline import get_m

def sample_flair_smooth_features(subject_id, subjects_dir, verbose=False):
    #TODO: rename function

    os.makedirs(opj(subjects_dir, subject_id, "surf_meld"), exist_ok=True)

    create_identity(opj(subjects_dir, subject_id))

    hemispheres = ["rh", "lh"]

    if os.path.isfile(opj(subjects_dir, subject_id, "mri", "FLAIR.mgz")):
        ds_features_to_generate = []
        dswm_features_to_generate = []
        for h in hemispheres:
            for d in [0.5, 0.25, 0.75, 0]:
                if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{h}.gm_FLAIR_{d}.mgh"):
                    ds_features_to_generate.append((h, d))
            for dwm in [0.5, 1]:
                if not os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/{h}.wm_FLAIR_{dwm}.mgh"):
                    dswm_features_to_generate.append((h, dwm))

        print(get_m(f'Sample FLAIR features : {ds_features_to_generate}', subject_id, 'INFO'))
        for dsf in ds_features_to_generate:
            # sampling volume to surface
            hemi = dsf[0]
            d = dsf[1]
            command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject_id}/mri/FLAIR.mgz --out {subjects_dir}/{subject_id}/surf_meld/{hemi}.gm_FLAIR_{d}.mgh --hemi {hemi} --projfrac {d} --srcreg {subjects_dir}/{subject_id}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject_id} --surf white"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

        print(get_m(f'Sample FLAIR features : {dswm_features_to_generate}', subject_id, 'INFO'))
        # Sample FLAIR 0.5mm and 1mm subcortically & smooth using 10mm Gaussian kernel
        for dswmf in dswm_features_to_generate:
            hemi = dswmf[0]
            dwm = dswmf[1]
            command = f"SUBJECTS_DIR='' mri_vol2surf --src {subjects_dir}/{subject_id}/mri/FLAIR.mgz --out {subjects_dir}/{subject_id}/surf_meld/{hemi}.wm_FLAIR_{dwm}.mgh --hemi {hemi} --projdist -{dwm} --srcreg {subjects_dir}/{subject_id}/mri/transforms/Identity.dat --trgsubject {subjects_dir}/{subject_id} --surf white"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False
    else:
        print(get_m(f'No FLAIR.mgh found. Skip sampling FLAIR feature', subject_id, 'INFO'))

    
    for hemi in hemispheres:
        # Calculate curvature
        print(get_m(f'Calculate curvatures', subject_id, 'INFO'))

        command = f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f white -g --writeCurvatureFiles {subject_id} {hemi} curv"
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode==0:
            pass
        else:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        command = f"SUBJECTS_DIR={subjects_dir} mris_curvature_stats -f pial -g --writeCurvatureFiles {subject_id} {hemi} curv"
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        # Convert mean curvature and sulcal depth to .mgh file type
        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.curv {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.curv.mgh"
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.sulc {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.sulc.mgh"
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.pial.K.crv {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K.mgh"
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        # get gaussian curvature
        print(get_m(f'Compute gaussian curvature', subject_id, 'INFO'))
        input = opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.pial.K.mgh")
        output = opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.pial.K_filtered.mgh")
        demo = nb.load(input)
        curvature = io_meld.load_mgh(input)
        curvature = np.absolute(curvature)
        curvature = np.clip(curvature, 0, 20)
        io_meld.save_mgh(output, curvature, demo)
        command = f"SUBJECTS_DIR={subjects_dir} mris_fwhm --s {subject_id} --hemi {hemi} --cortex --smooth-only --fwhm 20 --i {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K_filtered.mgh --o {subjects_dir}/{subject_id}/surf_meld/{hemi}.pial.K_filtered.sm20.mgh"
        # proc = Popen(command, shell=True, stdout = DEVNULL, stderr=STDOUT)
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False

        # get thickness
        print(get_m(f'Get thickness and white-grey matter contrast', subject_id, 'INFO'))
        command = f"SUBJECTS_DIR={subjects_dir} mris_convert -c {subjects_dir}/{subject_id}/surf/{hemi}.thickness {subjects_dir}/{subject_id}/surf/{hemi}.white {subjects_dir}/{subject_id}/surf_meld/{hemi}.thickness.mgh"
        proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr= proc.communicate()
        if verbose:
            print(stdout)
        if proc.returncode!=0:
            print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
            return False


        shutil.copy(
            opj(subjects_dir, subject_id, "surf", f"{hemi}.w-g.pct.mgh"),
            opj(subjects_dir, subject_id, "surf_meld", f"{hemi}.w-g.pct.mgh"),
        )


if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='sample FLAIR and create curvatures')
    parser.add_argument('subject_id', type=str,
                        help='subject_id')
    parser.add_argument('subjects_dir', type=str,
                        help='freesurfer subject directory ')
    args = parser.parse_args()
    #save subjects dir and subject ids. import the text file containing subject ids
    subject_id=args.subject_id
    subjects_dir=args.subject_id
    sample_flair_smooth_features(subject_id, subjects_dir)