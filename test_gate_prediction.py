#%%
import os
import numpy as np
import tempfile
import pandas as pd
from meld_graph.paths import (
                              EXPERIMENT_PATH,
                              DEMOGRAPHIC_FEATURES_FILE
                              )
from meld_graph.evaluation import Evaluator
from meld_graph.experiment import Experiment
from meld_graph.meld_cohort import MeldCohort
from meld_graph.tools_pipeline import  create_dataset_file, create_demographic_file
import sys
sys.path.append("/Users/mathilderipart/Documents/work/scripts/meld_graph")

#%%
subject_ids = np.array(['MELD_H101_3T_C_00005', 'MELD_H101_3T_C_00007', 'MELD_H101_3T_C_00008', 'MELD_H101_3T_C_00011', 'MELD_H101_3T_FCD_00058', 'MELD_H101_3T_FCD_00040'])
output_dir = '/Users/mathilderipart/Documents/work/261022_MELD_graph_test_gate/meld_data/output/classifier_outputs/23-10-30_LVHZ_dcp/fold_all'
plot_images = True
saliency=False
experiment_path=os.path.join(EXPERIMENT_PATH, '23-10-30_LVHZ_dcp', 'fold_all')
hdf5_file_root= '{site_code}_{group}_featurematrix_combat_6_kernels_robustCombat_NewSite.hdf5'
harmo_code ='H101'

# create demographic file
demographic_file_tmp = DEMOGRAPHIC_FEATURES_FILE
df = pd.DataFrame()
if isinstance(subject_ids, str):
    subjects_ids=[subject_ids]
df['ID']=subject_ids.astype(str)
df['Harmo code']=[str(harmo_code) for subject in subject_ids]
df['Group']=['patient' if '_FCD_' in subject else 'control' for subject in subject_ids ]
df['Scanner']=['XT' for subject in subject_ids]
df.to_csv(demographic_file_tmp)


# create dataset csv
tmp = tempfile.NamedTemporaryFile(mode="w")
create_dataset_file(subject_ids, tmp.name)

# load models
exp = Experiment.from_folder(experiment_path)

#update experiment 
exp.cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=tmp.name)
exp.data_parameters["hdf5_file_root"] = hdf5_file_root
exp.data_parameters["dataset"] = tmp.name
exp.data_parameters["augment_data"] = {}
exp.experiment_path = experiment_path

# launch evaluation
cohort = MeldCohort(
                hdf5_file_root=exp.data_parameters["hdf5_file_root"],
                dataset=exp.data_parameters["dataset"],
            )
eva = Evaluator(
    experiment=exp,
    checkpoint_path=experiment_path,
    cohort=cohort,
    subject_ids=subject_ids,
    save_dir=output_dir,
    mode="test",
    model_name="best_model",
    threshold='slope_threshold',
    thresh_and_clust=True,
    saliency=saliency,
    make_images=plot_images,
)
#%%
#predict for the dataset
eva.load_predict_data(
    save_prediction=False,
    roc_curves_thresholds=None,
    )

#%%
#threshold predictions
eva.threshold_and_cluster()

#%%
#write results in csv
eva.stat_subjects()

#%%
# gate distance thresholding 
threshold_gate = 0.48
data_dictionary=eva.data_dictionary

# if data_dictionary is None:
#     data_dictionary = self.data_dictionary
for subj_id, data in data_dictionary.items():
    prediction_clustered = data['cluster_thresholded']
    prediction_dist =  data['distance_map']

    clusters = set(prediction_clustered)
    clusters = [cl for cl in clusters if cl!=0]
    print('Number clusters before gate thresholding: ', len(clusters))

    for cl in clusters:
        dist_cl = prediction_dist[prediction_clustered==cl]
        dist_cl_median = np.median(dist_cl)
        print(dist_cl_median)
        if dist_cl_median >=0.48:
            print('gating')
            prediction_clustered[prediction_clustered==cl] = 0

    clusters = set(prediction_clustered)
    clusters = [cl for cl in clusters if cl!=0]
    print('Number clusters after gate thresholding: ', len(clusters))

    eva.data_dictionary[subj_id]['cluster_thresholded'] = prediction_clustered

#%%
#write results in csv
eva.stat_subjects()

#%%
#plot images 
if plot_images: 
    eva.plot_subjects_prediction()
# #compute saliency:
# if saliency:
#     eva.calculate_saliency()
# %%
