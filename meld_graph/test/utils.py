import os
import h5py
import numpy as np
import pandas as pd
from meld_graph.paths import BASE_PATH, DEFAULT_HDF5_FILE_ROOT, MELD_DATA_PATH, DEMOGRAPHIC_FEATURES_FILE
from meld_graph.tools_pipeline import create_dataset_file

print(DEMOGRAPHIC_FEATURES_FILE)

def create_test_demos():
    data = {"ID":{"0":"sub-test001","1":"MELD_TEST_15T_C_0001","2":"MELD_TEST_15T_C_0002","3":"MELD_TEST_15T_C_0003","4":"MELD_TEST_15T_C_0004","5":"MELD_TEST_15T_C_0005","6":"MELD_TEST_3T_C_0001","7":"MELD_TEST_3T_C_0002","8":"MELD_TEST_3T_C_0003","9":"MELD_TEST_3T_C_0004","10":"MELD_TEST_3T_C_0005","11":"MELD_TEST_15T_FCD_0002","12":"MELD_TEST_15T_FCD_0003","13":"MELD_TEST_15T_FCD_0004","14":"MELD_TEST_15T_FCD_0005","15":"MELD_TEST_15T_FCD_0006","16":"MELD_TEST_3T_FCD_0002","17":"MELD_TEST_3T_FCD_0003","18":"MELD_TEST_3T_FCD_0004","19":"MELD_TEST_3T_FCD_0005","20":"MELD_TEST_3T_FCD_0006"},
    "Harmo code":{"0":"TEST","1":"TEST","2":"TEST","3":"TEST","4":"TEST","5":"TEST","6":"TEST","7":"TEST","8":"TEST","9":"TEST","10":"TEST","11":"TEST","12":"TEST","13":"TEST","14":"TEST","15":"TEST","16":"TEST","17":"TEST","18":"TEST","19":"TEST","20":"TEST"},
    "Group ":{"0":"patient","1":"control","2":"control","3":"control","4":"control","5":"control","6":"control","7":"control","8":"control","9":"control","10":"control","11":"patient","12":"patient","13":"patient","14":"patient","15":"patient","16":"patient","17":"patient","18":"patient","19":"patient","20":"patient"},
    "Age at preoperative":{"0":25,"1":7,"2":9,"3":14,"4":3,"5":15,"6":26,"7":22,"8":4,"9":5,"10":12,"11":4,"12":6,"13":20,"14":12,"15":7,"16":4,"17":6,"18":10,"19":9,"20":12},
    "Sex":{"0":1,"1":0,"2":1,"3":0,"4":0,"5":1,"6":0,"7":1,"8":0,"9":0,"10":1,"11":0,"12":1,"13":1,"14":0,"15":1,"16":0,"17":0,"18":1,"19":0,"20":0},
    "Scanner":{"0":"3T","1":"15T","2":"15T","3":"15T","4":"15T","5":"15T","6":"3T","7":"3T","8":"3T","9":"3T","10":"3T","11":"15T","12":"15T","13":"15T","14":"15T","15":"15T","16":"3T","17":"3T","18":"3T","19":"3T","20":"3T"},}

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(MELD_DATA_PATH, DEMOGRAPHIC_FEATURES_FILE))

    create_dataset_file(df['ID'].values, '/tmp/dataset_test.csv')

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