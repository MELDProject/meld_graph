#### tests for dataset.py (Dataset class) ####
# tested functions:
#   load_combined_hemisphere_data
#   Dataset - behaviour with different flags, active selection
# NOTE:
#   these tests require a test dataset, that is created with get_test_data()
#   executing this function may take a while the first time (while the test data is being created)
# MISSING TESTS:
#   Dataset - test asserting correct handling of boundary zones in Dataset

from meld_graph.dataset import GraphDataset
from meld_graph.download_data import get_test_data
from meld_graph.meld_cohort import MeldSubject, MeldCohort
import pytest
from meld_graph.paths import NVERT, DEFAULT_HDF5_FILE_ROOT
# from meld_graph.network_tools import build_model
from meld_graph.test.utils import create_test_demos
import numpy as np
from copy import deepcopy


# @pytest.fixture(autouse=True)
# def setup_teardown_tests():
#     get_test_data()
#     yield


@pytest.fixture(scope="session")
def data_parameters():
    data_parameters = {
        "harmo code": ["TEST"],
        "scanners": ["15T", "3T"],
        "group": "patient",
        "features_to_exclude": [],
        "subject_features_to_exclude": ["FLAIR"],
        "fold_n": 0,
        "number_of_folds": 5,
        "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
        # --- params for Dataset class creation ---
        "batch_size": 8,
        "shuffle_each_epoch": True,
        "icosphere_parameters": {
            "distance_type": "exact",
            "combine_hemis": None,
            "conv_type": "SpiralConv",
            "icosphere_path":"data/icospheres/"
            },
        "augment_data": {
            "augment_lesion": {"p": 0.0},
            "blur": {"p": 0.2},
            "brightness": {"p": 0.15},
            "contrast": {"p": 0.15},
            "extend_lesion": {"p": 0.0},
            "flipping": {"file": "data/flipping/flipping_ico7_3.npy","p": 0.5},
            "gamma": {"p": 0.15},
            "low_res": {"p": 0.25},
            "noise": {"p": 0.15},
            "spinning": {"file": "data/spinning/spinning_ico7_10.npy","p": 0.2},
            "warping": {"file": "data/warping/warping_ico7_10.npy","p": 0.2},
            },
        "synthetic_data": {"run_synthetic": False,},
        "combine_hemis": None,
        "lesion_bias": 0,
        "lobes": False,
        "object_detection": True,
        "smooth_labels": False,
        "preprocessing_parameters": {
            "scaling": None,
            "zscore": "feature_means_nocombat.json"
            },
    }
    return data_parameters

# Dataset class tests
def test_dataset_flags(data_parameters):
    create_test_demos()
    c = MeldCohort(hdf5_file_root=data_parameters["hdf5_file_root"], dataset='/tmp/dataset_test.csv')

    subject_ids = c.get_subject_ids(**data_parameters)
    subject_ids = subject_ids[0:5]
    features_list = c.get_features(features_to_exclude=data_parameters["features_to_exclude"])

    # test contra flag
    # make sure that selected non-lesional data comes from contra hemisphere
    cur_data_params = dict(data_parameters, features=features_list)
    dataset = GraphDataset(subject_ids, cohort=c, params=cur_data_params)
    i=0
    for data in dataset:
        # check that data are there and are correct size
        i=i+1
        assert (data.x.shape[1]==len(features_list))
        assert (data.x.shape[0]==NVERT)
    assert i==len(subject_ids*2)