#define user paths for different user systems
import os
import pwd
from configparser import ConfigParser, NoOptionError, NoSectionError

# get scripts dir (parent dir of dir that this file is in)
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# read config file from scripts_dir
config_fname = os.path.join(SCRIPTS_DIR, 'meld_config.ini')
config = ConfigParser()
config.read(config_fname)

try:
    MELD_DATA_PATH = config.get('DEFAULT', 'meld_data_path')
    print(f'Setting MELD_DATA_PATH to {MELD_DATA_PATH}')
except NoOptionError as e:
    print(f'No meld_data_path defined in {config_fname}')
    MELD_DATA_PATH = ""

try:
    BASE_PATH = config.get('develop', 'base_path')
    print(f'Setting BASE_PATH to {BASE_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No base_path defined in {config_fname}!")
    BASE_PATH = ""
try:
    EXPERIMENT_PATH = config.get('develop', 'experiment_path')
    print(f'Setting EXPERIMENT_PATH to {EXPERIMENT_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No experiment_path defined in {config_fname}!")
    EXPERIMENT_PATH = ""
try:
    FS_SUBJECTS_PATH = config.get('develop', 'fs_subjects_path')
    print(f'Setting FS_SUBJECTS_PATH to {FS_SUBJECTS_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No fs_subjects_path defined in {config_fname}!")
    FS_SUBJECTS_PATH = ""
try:
    MELD_PARAMS_PATH = config.get('develop', 'meld_params_path')
    print(f'Setting BASE_PATH to {MELD_PARAMS_PATH}')
except (NoOptionError, NoSectionError) as e:
    print(f"No meld_params_path defined in {config_fname}!")
    MELD_PARAMS_PATH = ""

# paths to important data files - relative to BASE_PATH
COMBAT_PARAMS_FILE = 'Combat_parameters_6_kernels_robustCombat.hdf5'
NORM_CONTROLS_PARAMS_FILE = 'Norm_controls_parameters_6_kernels_robustCombat.hdf5'
# FINAL_SCALING_PARAMS = 'scaling_params_with0.json'
CLIPPING_PARAMS_FILE='clip_params_MELD.json'

# qc-ed demographic features
DEMOGRAPHIC_FEATURES_FILE = "demographics_qc_allgroups.csv"
# raw demographic features (participants.csv for each site)
DEMOGRAPHIC_FEATURES_RAW_FILE = os.path.join("MELD_{site_code}", "MELD_{site_code}_participants.csv")

CORTEX_LABEL_FILE = os.path.join("fsaverage_sym", "label", "lh.cortex.label")
SURFACE_FILE = os.path.join("fsaverage_sym", "surf", "lh.sphere")
SURFACE_PARTIAL= os.path.join("fsaverage_sym","surf","lh.partial_inflated")
BOUNDARY_ZONE_FILE = os.path.join("boundary_zones", "lesion_borderzones.hdf5")
DK_ATLAS_FILE = os.path.join("fsaverage_sym", "label", "lh.aparc.annot")
SMOOTH_CALIB_FILE = os.path.join("fsaverage_sym", "surf", "lh.pial")

# default values
# filename of hdf5 files
DEFAULT_HDF5_FILE_ROOT = "{site_code}_{group}_featurematrix_combat.hdf5"
#dataset
MELD_DATASET = "MELD_dataset_V6.csv"
NEWSUBJECTS_DATASET = "MELD_dataset_newSubjects.csv"
# number of vertices per hemi
NVERT = 163842

# pre-trained model path and name
MODEL_PATH = '23-10-30_MSBS_dcop_with_combat/fold_all'
MODEL_NAME = 'best_model'

#list of MELD sites
MELD_SITE_CODES = [
    "H2","H3","H4","H5","H6","H7","H9","H10","H11","H12","H14","H15","H16","H17","H18","H19","H21","H23","H24","H26",
]

# username = pwd.getpwuid(os.getuid())[0]
# username = 'kw350'
# # get scripts dir (parent dir of dir that this file is in)
# SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# for EXPERIMENT_PATH in [
#     "/Users/hannah.spitzer/projects/MELD/experiments_graph",  # Hannah's local experiment folder
#     "/rds/project/kw350/rds-kw350-meld/experiments_graph/{}".format(username),  # user-specific rds experiment folder
#     "/home/kw350/software/gdl/meld_classifier_GDL/scripts/",
# ]:
#     if os.path.exists(EXPERIMENT_PATH):
#         print("Setting EXPERIMENT_PATH to " + EXPERIMENT_PATH)
#         break
# if not os.path.exists(EXPERIMENT_PATH):
#     print('WARNING: EXPERIMENT_PATH not found, setting to "", need to add it to paths.py')
#     EXPERIMENT_PATH = ""


# def load_config(config_file):
#     """load config.py file and return config object"""
#     import importlib.machinery, importlib.util

#     loader = importlib.machinery.SourceFileLoader("config", config_file)
#     spec = importlib.util.spec_from_loader(loader.name, loader)
#     config = importlib.util.module_from_spec(spec)
#     loader.exec_module(config)
#     return config
