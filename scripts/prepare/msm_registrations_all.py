from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import numpy as np

import os
import subprocess
import nibabel as nb
import meld_classifier.paths as p
import argparse


def save_numpy_array_as_func_gii(arr, filename):

    """
    Save a NumPy array as a func.gii file.

    Parameters:
    arr: NumPy array to be saved.
    filename: Name of the file to save the array in.

    Returns:
    None
    """
    # Create a Nifti1Image object
    img = nb.gifti.GiftiImage()
    data = nb.gifti.GiftiDataArray(arr)

    # Set the data array in the image
    img.add_gifti_data_array(data)

    # Save the image as a func.gii file
    nb.save(img, filename)


def fit_registration(sub_id, hemi):
    print(f"registering {sub_id}")
    msm_bin = "/home/kw350/software/msm_centos_v3"
    inmesh = os.path.join(p.MELD_DATA_PATH, "fsaverage_sym", "surf", "lh.sphere.surf.gii")
    refmesh = inmesh
    indata = os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", f"{hemi}.{sub_id}.txt")
    refdata = os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", f"{hemi}.template.txt")
    outname = os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", f"{hemi}_{sub_id}")
    config = os.path.join(p.MELD_DATA_PATH, "msm", "config")

    cmd = f"{msm_bin}  --inmesh={inmesh} --refmesh={refmesh} --indata={indata} --refdata={refdata} -o {outname}"
    print(cmd)
    subprocess.call(cmd, shell=True)
    return


def write_funci_gii(sub_id, hemi, subj):
    if subj.has_flair:
        features = [
            ".combat.on_lh.pial.K_filtered.sm20.mgh",
            ".combat.on_lh.thickness.sm3.mgh",
            ".combat.on_lh.thickness_regression.sm3.mgh",
            ".combat.on_lh.sulc.sm3.mgh",
            ".combat.on_lh.curv.sm3.mgh",
            ".combat.on_lh.w-g.pct.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ]
    else:
        features = [
            ".combat.on_lh.pial.K_filtered.sm20.mgh",
            ".combat.on_lh.thickness.sm3.mgh",
            ".combat.on_lh.thickness_regression.sm3.mgh",
            ".combat.on_lh.sulc.sm3.mgh",
            ".combat.on_lh.curv.sm3.mgh",
            ".combat.on_lh.w-g.pct.sm3.mgh",
        ]
    hemi_data = subj.load_features_values(features, hemi=hemi)
    save_numpy_array_as_func_gii(
        hemi_data,
        os.path.join(p.MELD_DATA_PATH, "msm", "msm_input_data", f"{hemi}.{sub_id}.func.gii"),
    )
    return features


def write_lesion_mask_gii(sub_id, hemi, subj):

    _, lesion_mask = subj.load_feature_lesion_data([""], hemi=hemi)
    save_numpy_array_as_func_gii(
        lesion_mask.astype(np.int32),
        os.path.join(p.MELD_DATA_PATH, "msm", "msm_input_data", f"{hemi}.{sub_id}.label.gii"),
    )
    return


def register_surface_data_over(hemi, sub_id):
    wb_bin = "/home/kw350/software/workbench/bin_rh_linux64/wb_command"
    current_sphere = os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", f"{hemi}_{sub_id}sphere.reg.surf.gii")
    template = os.path.join(p.MELD_DATA_PATH, "fsaverage_sym", "surf", "lh.sphere.surf.gii")
    metric_in = os.path.join(p.MELD_DATA_PATH, "msm", "msm_input_data", f"{hemi}.{sub_id}.func.gii")
    metric_out = os.path.join(p.MELD_DATA_PATH, "msm", "msm_output_data", f"{hemi}.{sub_id}.func.gii")
    cmd = f"{wb_bin} -metric-resample {metric_in} {current_sphere} {template} BARYCENTRIC {metric_out}"
    subprocess.call(cmd, shell=True)
    return


def register_label_data_over(hemi, sub_id):
    wb_bin = "/home/kw350/software/workbench/bin_rh_linux64/wb_command"
    current_sphere = os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", f"{hemi}_{sub_id}sphere.reg.surf.gii")
    template = os.path.join(p.MELD_DATA_PATH, "fsaverage_sym", "surf", "lh.sphere.surf.gii")
    metric_in = os.path.join(p.MELD_DATA_PATH, "msm", "msm_input_data", f"{hemi}.{sub_id}.label.gii")
    metric_out = os.path.join(p.MELD_DATA_PATH, "msm", "msm_output_data", f"{hemi}.{sub_id}.label.gii")
    cmd = f"{wb_bin} -label-resample {metric_in} {current_sphere} {template} BARYCENTRIC {metric_out}"
    subprocess.call(cmd, shell=True)
    return


def read_and_write_to_hdf5(hemi, sub_id, subj, hdf5_file_out, features):
    metric_out = os.path.join(p.MELD_DATA_PATH, "msm", "msm_output_data", f"{hemi}.{sub_id}.func.gii")
    registered_files = nb.load(metric_out)
    stacked = np.vstack(registered_files.agg_data())

    for fi, feature in enumerate(features):
        print(feature)
        done = False
        while not done:
            try:
                subj.write_feature_values(
                    feature,
                    stacked[fi, cohort.cortex_mask],
                    hemis=[hemi],
                    hdf5_file_root=hdf5_file_out,
                )
                done = True
            except BlockingIOError:
                pass
    return


def label_to_hdf5(hemi, sub_id, subj, hdf5_file_out):
    metric_out = os.path.join(p.MELD_DATA_PATH, "msm", "msm_output_data", f"{hemi}.{sub_id}.label.gii")
    registered_files = nb.load(metric_out)
    stacked = registered_files.agg_data()
    done = False
    while not done:
        try:
            subj.write_feature_values(
                ".on_lh.lesion.mgh",
                stacked[cohort.cortex_mask],
                hemis=[hemi],
                hdf5_file_root=hdf5_file_out,
            )
            done = True
        except BlockingIOError:
            pass
    return


def tidy_up(sub_id, hemi):
    metric_in = os.path.join(p.MELD_DATA_PATH, "msm", "msm_input_data", f"{hemi}.{sub_id}.func.gii")
    metric_out = os.path.join(p.MELD_DATA_PATH, "msm", "msm_output_data", f"{hemi}.{sub_id}.func.gii")
    os.remove(metric_in)
    os.remove(metric_out)
    return


subjects = np.loadtxt(os.path.join(p.MELD_DATA_PATH, "msm", "subject_ids.txt"), dtype=str)
for sub_id in subjects:
    dataset = "MELD_dataset_V6.csv"
    hdf5_file_root = "{site_code}_{group}_featurematrix_combat_6_kernels.hdf5"
    hdf5_file_root_out = "{site_code}_{group}_featurematrix_combat_msm.hdf5"
    cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=dataset)
    subj = MeldSubject(sub_id, cohort=cohort)
    print(sub_id)
    if subj.has_flair:
        features = [
            ".combat.on_lh.pial.K_filtered.sm20.mgh",
            ".combat.on_lh.thickness.sm3.mgh",
            ".combat.on_lh.thickness_regression.sm3.mgh",
            ".combat.on_lh.sulc.sm3.mgh",
            ".combat.on_lh.curv.sm3.mgh",
            ".combat.on_lh.w-g.pct.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ]
    else:
        features = [
            ".combat.on_lh.pial.K_filtered.sm20.mgh",
            ".combat.on_lh.thickness.sm3.mgh",
            ".combat.on_lh.thickness_regression.sm3.mgh",
            ".combat.on_lh.sulc.sm3.mgh",
            ".combat.on_lh.curv.sm3.mgh",
            ".combat.on_lh.w-g.pct.sm3.mgh",
        ]
    for hemi in ["rh", "lh"]:
        # fit_registration(sub_id,hemi)
        #  features = write_funci_gii(sub_id,hemi,subj)
        #  register_surface_data_over(hemi,sub_id)
        read_and_write_to_hdf5(hemi, sub_id, subj, hdf5_file_root_out, features)
        if subj.has_lesion() and subj.get_lesion_hemisphere() == hemi:
            #   write_lesion_mask_gii(sub_id,hemi,subj)
            #     register_label_data_over(hemi,sub_id)
            label_to_hdf5(hemi, sub_id, subj, hdf5_file_root_out)

    # function to map lesion mask
    # tidy up
    # tidy_up(sub_id,hemi)
