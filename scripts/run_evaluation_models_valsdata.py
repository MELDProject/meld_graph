import os
import meld_graph.experiment
import meld_graph.evaluation
from meld_graph.dataset import GraphDataset
from meld_classifier.meld_cohort import MeldCohort
from meld_graph.paths import EXPERIMENT_PATH
from meld_graph.evaluation import Evaluator
import numpy as np


### DEFINE MODELS TO RUN

# initialise models you want to run
EXPERIMENT_PATH = "/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350"

# defining paths - change both fname and model_base_path
fname = "22-12-14_GXRU_dice_ce_fold"
# fname = '22-12-14_GXRU_dcd_maew_fold'
model_base_paths_roots = os.path.join(EXPERIMENT_PATH, fname, "s_2")
model_base_paths = {}
for fold in np.arange(10):
    model_base_paths[f"dc_{fold}"] = os.path.join(model_base_paths_roots, f"fold_0{fold}")


### initialise saving outputs
use_preload_dataset = True
# NOTE: if output path is None, will save results in each experiment folder
output_path = None
# output_path = '/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/22-12-15_trainval'
# output_path = '/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/22-12-15_test'

### Create dataset with vals and train data

model_base_path = model_base_paths[list(model_base_paths.keys())[0]]
checkpoint_path = os.path.join(EXPERIMENT_PATH, model_base_path)
exp = meld_graph.experiment.Experiment.from_folder(checkpoint_path)

#
# subjects = exp.data_parameters['train_ids']+ exp.data_parameters['val_ids']
subjects = exp.data_parameters["test_ids"]
cohort = MeldCohort(
    hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5",
    dataset="MELD_dataset_V6.csv",
)

if use_preload_dataset:
    features = exp.data_parameters["features"]

    # initiate params
    params = {
        "features": features,
        "augment_data": {},
        "lesion_bias": 0,
        "lobes": False,
        "synthetic_data": {
            "run_synthetic": False,
        },
        "number_of_folds": 10,
        "preprocessing_parameters": {
            "scaling": None,
            "zscore": "../data/feature_means.json",
        },
        "combine_hemis": None,
    }

    # load dataset
    dataset = GraphDataset(subjects, cohort, params, mode="test")
    cohort = None
    subjects = None
else:
    dataset = None
### LOAD MODELS AND PREDICT

for model_name in model_base_paths.keys():

    # load experiment already trained using checkpoint path
    model_base_path = model_base_paths[model_name]
    checkpoint_path = os.path.join(EXPERIMENT_PATH, model_base_path)
    exp = meld_graph.experiment.Experiment.from_folder(checkpoint_path)

    # Run the evaluation on the test data and save into directory provided
    if output_path is None:
        save_dir = None
    else:
        save_dir = os.path.join(output_path, model_name)

    eva = Evaluator(
        experiment=exp,
        checkpoint_path=checkpoint_path,
        save_dir=save_dir,
        make_images=True,
        dataset=dataset,
        cohort=cohort,
        subject_ids=subjects,
        mode="test",
    )

    # load data and predict
    print("loading predicting")
    eva.load_predict_data(store_predictions=False)
    # calculate stats
# eva.stat_subjects()
#  make images
# eva.plot_subjects_prediction()
