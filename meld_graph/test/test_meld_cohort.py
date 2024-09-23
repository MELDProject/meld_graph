#### tests for MeldCohort ####
# tested functions:
#   full_feature_list (2 tests)
#   get_subject_ids
#   get_sites
#   split_hemispheres
#   cortex_label attribute
# NOTE:
#   these tests require a test dataset, that is created with get_test_data()
#   executing this function may take a while the first time (while the test data is being created)
# MISSING TESTS:
#   test getting / filtering features?

import pytest
from meld_graph.meld_cohort import MeldCohort, MeldSubject
from meld_graph.paths import NVERT, BASE_PATH, DEFAULT_HDF5_FILE_ROOT
import numpy as np
import warnings
from meld_graph.download_data import get_test_data
import tempfile
import pandas as pd
from meld_graph.test.utils import create_test_demos


create_test_demos()

@pytest.mark.parametrize(
    "hdf5_file_root", [DEFAULT_HDF5_FILE_ROOT]
)
def test_features_overlap_with_list(hdf5_file_root):
    """test if computed full_feature_list overlaps with manual feature list"""
    c = MeldCohort(hdf5_file_root=hdf5_file_root, dataset='/tmp/dataset_test.csv')
    if len(c.get_subject_ids()) == 0:
        warnings.warn("hdf5_file_root {hdf5_file_root} does not seem to exist on this system. Skipping this test.")
        return
    combat = "combat" in hdf5_file_root
    if combat:
        # insert combat feature list here.
        full_feature_list = [
        ".combat.on_lh.pial.K_filtered.sm20.mgh",
        ".combat.on_lh.thickness.sm3.mgh",
        ".combat.on_lh.thickness_regression.sm3.mgh",
        ".combat.on_lh.w-g.pct.sm3.mgh",
        ".combat.on_lh.sulc.sm3.mgh",
        ".combat.on_lh.curv.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.sm3.mgh",
        ".combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
        ".combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.thickness.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.sulc.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.curv.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.thickness.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.sulc.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.curv.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh"
        ]
    else:
        full_feature_list = [
            ".on_lh.curv.mgh",
            ".on_lh.gm_FLAIR_0.25.mgh",
            ".on_lh.gm_FLAIR_0.5.mgh",
            ".on_lh.gm_FLAIR_0.75.mgh",
            ".on_lh.gm_FLAIR_0.mgh",
            ".on_lh.pial.K_filtered.sm20.mgh",
            ".on_lh.sulc.mgh",
            ".on_lh.thickness.mgh",
            ".on_lh.w-g.pct.mgh",
            ".on_lh.wm_FLAIR_0.5.mgh",
            ".on_lh.wm_FLAIR_1.mgh",
        ]
    diff = set(c.full_feature_list).difference(full_feature_list)
    assert len(diff) == 0, f"features {diff} are not in both lists"


def test_features_consistent():
    """test that all subjects in cohort have same features"""
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset='/tmp/dataset_test.csv')
    features = c.full_feature_list
    for subj_id in c.get_subject_ids():
        for hemi in ["lh", "rh"]:
            subj_features = MeldSubject(subj_id, c).get_feature_list(hemi=hemi)
            diff = set(subj_features).difference(features)
            assert len(diff) == 0, f"{subj_id}: features {diff} are not overlapping"


def test_get_sites():
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset='/tmp/dataset_test.csv')
    # TEST must be in sites, because we just created test data
    assert "TEST" in c.get_sites()


def test_cortex_label():
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset='/tmp/dataset_test.csv')
    # check that cortex is ordered list
    assert ((c.cortex_label[1:] - c.cortex_label[:-1]) > 0).all()


def test_get_subject_ids():
    """
    tests MeldCohort.get_subject_ids
    ensure that returned subjects are filtered correctly
    """
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset='/tmp/dataset_test.csv')

    # test site_codes flag
    all_subjects = c.get_subject_ids(site_codes="TEST")
    for subj_id in all_subjects:
        assert MeldSubject(subj_id, cohort=c).site_code == "TEST"

    # test group flag
    patients = c.get_subject_ids(site_codes="TEST", group="patient")
    for subj_id in patients:
        assert MeldSubject(subj_id, cohort=c).group == "patient"
    controls = c.get_subject_ids(site_codes="TEST", group="control")
    assert len(all_subjects) == len(patients) + len(controls)

    # test subject_features_to_exclude flag
    flair_subjects = c.get_subject_ids(site_codes="TEST", subject_features_to_exclude=["FLAIR"])
    _, flair_features = c._filter_features(features_to_exclude=["FLAIR"], return_excluded=True)
    for subj_id in flair_subjects:
        # does this subject have flair features?
        assert np.any(MeldSubject(subj_id, cohort=c).load_feature_values(flair_features[0]) != 0)

    # test scanners flag
    subjects_3T = c.get_subject_ids(site_codes="TEST", scanners="3T")
    for subj_id in subjects_3T:
        assert MeldSubject(subj_id, cohort=c).scanner == "3T"
    subjects_15T = c.get_subject_ids(site_codes="TEST", scanners="15T")
    assert len(all_subjects) == len(subjects_3T) + len(subjects_15T)


def test_get_subject_ids_with_dataset():
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT,  dataset='/tmp/dataset_test.csv')
    ds_site_codes = ["TEST"]

    # create temp dataset file
    listids = c.get_subject_ids(site_codes=ds_site_codes, group="both")
    print(listids)
    df = pd.DataFrame({"subject_id": listids, "split": ["trainval" for _ in listids]})
    with tempfile.NamedTemporaryFile(dir=BASE_PATH) as dataset_file:
        df.to_csv(dataset_file.name)

        # get subjects using this dataset file
        c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset=dataset_file.name)
        subject_ids = c.get_subject_ids(group="both")
        subjects_in_dataset, _, _ = c.read_subject_ids_from_dataset()

        # check if subjects returned by get_subject_ids are a subset of the subjects defined in temp dataset
        assert len(subject_ids) <= len(listids)
        assert np.in1d(subject_ids, listids).all()
        assert np.in1d(listids, subjects_in_dataset).all()


def test_split_hemispheres():
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset='/tmp/dataset_test.csv')
    # test that splits data correctly
    # create data with ones for left hemi, twoes for right hemi
    input_data = np.concatenate([np.ones(len(c.cortex_label)), np.ones(len(c.cortex_label)) * 2])
    hemi_data = c.split_hemispheres(input_data)
    assert np.all(hemi_data["left"][c.cortex_label] == 1)
    assert np.all(hemi_data["right"][c.cortex_label] == 2)
    assert len(hemi_data["left"]) == NVERT
    assert len(hemi_data["right"]) == NVERT

    # test that raises an error when presented data of wrong shape
    input_data = np.zeros(100)
    with pytest.raises(AssertionError):
        c.split_hemispheres(input_data)
