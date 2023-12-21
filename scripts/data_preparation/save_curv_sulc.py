from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import numpy as np
import meld_classifier.paths as p
import nibabel as nb
import os
import subprocess

dataset = "MELD_dataset_V6.csv"
hdf5_file_root = "{site_code}_{group}_featurematrix_combat_6_kernels.hdf5"
cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=dataset)


def save_numpy_array_as_func_gii(filename, arr):
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


print("subject ids")
subjects = cohort.get_subject_ids(group="both")
all_features = np.zeros((len(subjects), len(cohort.cortex_mask)), dtype=np.float32)
template = np.zeros(len(cohort.cortex_mask), dtype=np.float32)
for hi, hemi in enumerate(["lh", "rh"]):
    for si, sub_id in enumerate(subjects):
        if si % 100 == 0:
            print(si)
        subj = MeldSubject(sub_id, cohort=cohort)
        features = [
            ".combat.on_lh.sulc.sm3.mgh",
            # '.combat.on_lh.curv.sm3.mgh',
        ]
        hemi_data = subj.load_features_values(features, hemi=hemi).astype(np.float32)
        all_features[si] = hemi_data.ravel()
        #  save_numpy_array_as_func_gii(os.path.join(p.MELD_DATA_PATH,'msm','msm_regs',f'{hemi}.{sub_id}.func.gii' ),hemi_data)
        np.savetxt(
            os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", f"{hemi}.{sub_id}.txt"),
            hemi_data,
        )

    template += np.mean(all_features, axis=0) / 2
# save_numpy_array_as_func_gii(os.path.join(p.MELD_DATA_PATH,'msm','msm_regs',f'{hemi}.template.func.gii'),template,)

np.savetxt(
    os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", "lh.template.txt"),
    template,
)
np.savetxt(
    os.path.join(p.MELD_DATA_PATH, "msm", "msm_regs", "rh.template.txt"),
    template,
)


np.savetxt(os.path.join(p.MELD_DATA_PATH, "msm", "subject_ids.txt"), subjects, fmt="%s")
