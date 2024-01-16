import os
import h5py
import numpy as np
from meld_graph.paths import BASE_PATH, DEFAULT_HDF5_FILE_ROOT

def create_test_data():
    """
    This function was initially used to create the random test dataset. 
    Now, the downloadable test data is preferred over recreating it.
    """
    from shutil import copyfile
    # edit copied H4 dataset to contain 0,1,2.... as feature values. 5 patients, 5 subjects
    # start from copies of H4 site
    print("creating test dataset MELD_TEST from MELD_H4...")
    test_data_dir = os.path.join(BASE_PATH, "MELD_TEST")
    os.makedirs(test_data_dir, exist_ok=True)
    for group in ["control", "patient"]:
        with h5py.File(os.path.join(BASE_PATH, "MELD_H4", DEFAULT_HDF5_FILE_ROOT.format(site_code='H4', group=group)), "r") as f_ref:
            with h5py.File(os.path.join(test_data_dir, DEFAULT_HDF5_FILE_ROOT.format(site_code='TEST', group=group)), "w") as f:
                for scanner in ["3T", "15T"]:
                    for i, old_patient_id in enumerate(f_ref["H4"][scanner][group].keys()):
                        if i < 5:
                            new_patient_id = old_patient_id.replace("H4", "TEST")
                            print("creating test data for {}".format(new_patient_id))
                            for hemi in ["lh", "rh"]:
                                for feature, value in f_ref["H4"][scanner][group][old_patient_id][hemi].items():
                                    dset = f.create_dataset(f"TEST/{scanner}/{group}/{new_patient_id}/{hemi}/{feature}", shape=value.shape, dtype=value.dtype)
                                    if feature == ".on_lh.lesion.mgh":
                                        dset[:] = value
                                    else:
                                        if hemi == "lh":
                                            dset[:] = np.arange(0, len(dset), dtype=np.float)
                                        else:
                                            dset[:] = np.arange(0, len(dset), dtype=np.float) + len(dset)
    return test_data_dir

if __name__ == '__main__':
    create_test_data()