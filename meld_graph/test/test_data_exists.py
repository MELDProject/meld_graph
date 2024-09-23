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
from meld_graph.test.utils import create_test_demos
from meld_graph.tools_pipeline import create_dataset_file
sites = ["TEST"]

#create demo tmp
create_test_demos()

@pytest.mark.data
@pytest.mark.parametrize("site", sites)
def test_cohort_exists(site):
    hdf5_file_root = "{site_code}_{group}_featurematrix_combat.hdf5"
    c = MeldCohort(hdf5_file_root=hdf5_file_root, dataset='/tmp/dataset_test.csv')
    # does exist at all?
    if len(c.get_subject_ids(group='all', site_codes=[site])) == 0:
        warnings.warn(f"hdf5_file_root {hdf5_file_root} does not exist on this system.")
        return
    patient_ids = c.get_subject_ids(group="patient", site_codes=[site])
    print(patient_ids)
    if len(patient_ids) == 0:
        warnings.warn(f"cohort for {hdf5_file_root} does not have patients for site {site}")
    control_ids = c.get_subject_ids(group="control", site_codes=[site])
    if len(control_ids) == 0:
        warnings.warn(f"cohort for {hdf5_file_root} does not have controls for site {site}")

@pytest.mark.data
@pytest.mark.parametrize("site", sites)
def test_borderzone_exists(site):
    hdf5_file_root = "{site_code}_{group}_featurematrix_combat.hdf5"
    c = MeldCohort(hdf5_file_root=hdf5_file_root, dataset='/tmp/dataset_test.csv')
    subject_ids = c.get_subject_ids(group="patient", lesional_only=True, site_codes=[site])
    print(subject_ids)
    # get a few random subject_ids to test if has borderzone
    for subj_id in np.random.default_rng().choice(subject_ids, size=min(len(subject_ids), 3), replace=False):
        subj = MeldSubject(subj_id, cohort=c)
        borderzone = subj.load_boundary_zone()
        # each patient should have a borderzone
        assert np.sum(borderzone) > 0, f"patient {subj_id} does not have a borderzone"
