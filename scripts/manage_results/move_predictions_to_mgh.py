import os
import tempfile
import numpy as np
import h5py
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import MELD_DATA_PATH
import nibabel as nb
import argparse
from meld_graph.tools_pipeline import get_m
from meld_graph.hdf5_utils import open_hdf5_file


def load_prediction(subject, hdf5, prediction_name="prediction_clustered"):
    results = {}
    with open_hdf5_file(hdf5, mode="r") as f:
        for hemi in ["lh", "rh"]:
            results[hemi] = f[subject][hemi][prediction_name][:]
    return results

def save_mgh(filename, array, demo):
    """save mgh file using nibabel and imported demo mgh file"""
    with tempfile.NamedTemporaryFile() as mmap_file:
        mmap = np.memmap(mmap_file.name, dtype="float32", mode="w+", shape=demo.get_data().shape)
        mmap[:, 0, 0] = array[:]
        output = nb.MGHImage(mmap, demo.affine, demo.header)
        nb.save(output, filename)

def move_predictions_to_mgh(subject_id, subjects_dir, prediction_file, output_dir, verbose=False):
    ''' move meld predictions from hdf to mgh freesurfer volume. Outputs are saved into freesurfer subject directory 
    inputs:
        subject_ids : subjects ID in an array
        subjects_dir : freesurfer subjects directory
        prediction_file : hdf5 file containing the MELD predictions
    '''
    c = MeldCohort()
    predictions = load_prediction(subject_id, prediction_file, prediction_name="cluster_thresholded_salient")
    for hemi in ["lh", "rh"]:
        prediction_h = predictions[hemi]
        overlay = np.zeros_like(c.cortex_mask, dtype=int)
        overlay[c.cortex_mask] = prediction_h
        try:
            demo = nb.load(os.path.join(subjects_dir, subject_id, "xhemi", "surf_meld", f"{hemi}.on_lh.thickness.mgh"))
        except:
            print(get_m(f'Could not load {os.path.join(subjects_dir, subject_id, "xhemi", "surf_meld", f"{hemi}.on_lh.thickness.mgh")} ', subject_id, 'ERROR')) 
            return False   
        save_dir=os.path.join(output_dir, subject_id, 'predictions')
        filename = os.path.join(save_dir, 'fsaverage_sym', f"{hemi}.prediction.mgh")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_mgh(filename, overlay, demo)
        

    
if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="create mgh file with predictions from hdf5 arrays")
    parser.add_argument(
        "--experiment-folder",
        help="Experiments folder",
    )
    parser.add_argument(
        "--experiment-name",
        help="subfolder to use, typically the ensemble model",
        default="ensemble_iteration",
    )
    parser.add_argument("--fold", default=None, help="fold number to use (by default all)")
    parser.add_argument(
        "--subjects_dir", default="", help="folder containing freesurfer outputs. It will store predictions there"
    )
    parser.add_argument("--list_ids", default=None, help="texte file containing list of ids to process")
    parser.add_argument("--output_dir", default="", help="folder to save predictions in mgh format")

    args = parser.parse_args()

    experiment_path = os.path.join(MELD_DATA_PATH, args.experiment_folder)
    subjects_dir = args.subjects_dir

    if args.fold == None: 
        prediction_file = os.path.join(
            experiment_path, "results", f"predictions_{args.experiment_name}.hdf5"
        )
    else : 
        prediction_file = os.path.join(
            experiment_path, f"fold_{args.fold}", "results", f"predictions_{args.experiment_name}.hdf5"
        )

    if args.list_ids:
        subjids = np.loadtxt(args.list_ids, dtype="str", ndmin=1)

    
    if os.path.isfile(prediction_file):
        for subject_id in subjids:
            move_predictions_to_mgh(subject_id, subjects_dir, prediction_file, args.output_dir)