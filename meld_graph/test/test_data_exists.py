#### tests for existing data on curreny system ####
# outputs warnings if a desired site does not exists
# TODO: update the tested sites / files with those that should exist
# tested files:
#   check if site data exists and contains patient ids
#   check if bordezone exists for each site / patient


from meld_graph.paths import DEFAULT_HDF5_FILE_ROOT
import pytest
from meld_graph.meld_cohort import MeldCohort, MeldSubject
import numpy as np
import warnings

sites = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H9",
    "H10",
    "H11",
    "H12",
    "H14",
    "H15",
    "H16",
    "H17",
    "H18",
    "H19",
    "H21",
    "H23",
    "H24",
    "H26",
]
hdf5_file_roots = ["{site_code}_{group}_featurematrix.hdf5", DEFAULT_HDF5_FILE_ROOT]

@pytest.mark.data
@pytest.mark.parametrize("hdf5_file_root", hdf5_file_roots)
def test_cohort_exists(hdf5_file_root):
    c = MeldCohort(hdf5_file_root=hdf5_file_root)
    # does exist at all?
    if len(c.get_subject_ids()) == 0:
        warnings.warn(f"hdf5_file_root {hdf5_file_root} does not exist on this system.")
        return
    for site in sites:
        patient_ids = c.get_subject_ids(group="patient", site_codes=[site])
        if len(patient_ids) == 0:
            warnings.warn(f"cohort for {hdf5_file_root} does not have patients for site {site}")
        control_ids = c.get_subject_ids(group="control", site_codes=[site])
        if len(control_ids) == 0:
            warnings.warn(f"cohort for {hdf5_file_root} does not have controls for site {site}")

@pytest.mark.data
@pytest.mark.parametrize("site", sites)
def test_borderzone_exists(site):
    c = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5")
    subject_ids = c.get_subject_ids(group="patient", lesional_only=True, site_codes=[site])
    # get a few random subject_ids to test if has borderzone
    for subj_id in np.random.default_rng().choice(subject_ids, size=min(len(subject_ids), 3), replace=False):
        subj = MeldSubject(subj_id, cohort=c)
        borderzone = subj.load_boundary_zone()
        # each patient should have a borderzone
        assert np.sum(borderzone) > 0, f"patient {subj_id} does not have a borderzone"
