# Script to create scaling parameters for the raw features

from meld_classifier.paths import BASE_PATH
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_graph.data_preprocessing import Preprocess
import os
import numpy as np

# define cohort to compute scaling parameters from

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
cohort = MeldCohort(
    hdf5_file_root="{site_code}_{group}_featurematrix.hdf5",
    dataset="MELD_dataset_V6.csv",
)

# define features to compute scaling parameters
features = [
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

# define scaling parameters file name
scaling_params_file = "scaling_params_GDL.json"

# create object preprocessing
scale = Preprocess(
    cohort,
    site_codes=site_codes,
    write_output_file=scaling_params_file,
    data_dir=BASE_PATH,
)

# compute scaling parameters
for feature in features:
    scale.compute_scaling_parameters(feature)
