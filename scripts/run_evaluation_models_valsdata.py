import os
import meld_graph.experiment
import meld_graph.evaluation
from meld_graph.dataset import GraphDataset
from meld_classifier.meld_cohort import MeldCohort
from meld_graph.paths import EXPERIMENT_PATH
from meld_graph.evaluation import Evaluator



### DEFINE MODELS TO RUN

# initialise models you want to run
EXPERIMENT_PATH='/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350'

model_base_paths = {
   'dcd': '/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/22-12-13_finetune/dcd/fold_00/',
   # 'dc': '/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/22-12-13_finetune/dc/fold_00/',
   # 'dcd_head': '/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/22-12-13_finetune/dcd_head/fold_00/',
}

### initialise saving outputs
use_preload_dataset = True 
output_path = '/rds/project/kw350/rds-kw350-meld/experiments_graph/co-ripa1/22-12-13_evaluation_real_valsdata_test'

### Create dataset with vals data

model_base_path=model_base_paths['dcd']
checkpoint_path=os.path.join(EXPERIMENT_PATH, model_base_path)
exp = meld_graph.experiment.Experiment.from_folder(checkpoint_path)

subjects = exp.data_parameters['val_ids']
cohort= MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_6.hdf5', dataset='MELD_dataset_V6.csv')

if use_preload_dataset:  
    features= exp.data_parameters['features']

    # initiate params
    params = {                
            "features":features,
            "augment_data": {}, 
            "lesion_bias":0,
            "lobes": False,
            "synthetic_data": {
                'run_synthetic': False,
                },
            "number_of_folds": 10,
            "preprocessing_parameters": {
                "scaling": None, 
                "zscore": '../data/feature_means.json',
                },
            "combine_hemis": None,
            }

    #load dataset
    dataset = GraphDataset(subjects, cohort, params, mode='test')
    cohort = None
    subjects = None
else:
    dataset = None
### LOAD MODELS AND PREDICT  

for model_name in model_base_paths.keys(): 
    
    #load experiment already trained using checkpoint path
    model_base_path=model_base_paths[model_name]
    checkpoint_path=os.path.join(EXPERIMENT_PATH, model_base_path)
    exp = meld_graph.experiment.Experiment.from_folder(checkpoint_path)

    # Run the evaluation on the test data and save into directory provided
    save_dir=os.path.join(output_path, model_name)

    eva = Evaluator(experiment = exp,
                    checkpoint_path = checkpoint_path,
                    save_dir = save_dir ,
                    make_images = True,
                    dataset=dataset,
                    cohort=cohort,
                    subject_ids = subjects,
                    mode = 'test'
                )

    # load data and predict
    eva.load_predict_data()
    # calculate stats 
    eva.stat_subjects()
    #  make images 
    eva.plot_subjects_prediction()

    