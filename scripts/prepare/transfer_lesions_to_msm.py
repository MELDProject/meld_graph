from meld_classifier.meld_cohort import MeldCohort
from meld_classifier.data_preprocessing import Preprocess
from meld_classifier.paths import BASE_PATH

site_codes = [
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

hdf5_file_root = "{site_code}_{group}_featurematrix_combat_6_kernels.hdf5"
hdf5_file_root_out = "{site_code}_{group}_featurematrix_combat_msm.hdf5"
dataset = "MELD_dataset_V6.csv"


c_smooth = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=dataset)
# create object msm
msm = Preprocess(
    c_smooth,
    site_codes=site_codes,
    write_hdf5_file_root=hdf5_file_root_out,
    data_dir=BASE_PATH,
)

# Transfer lesions in the new hdf5
msm.transfer_lesion()
