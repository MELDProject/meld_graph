import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from meld_classifier.meld_cohort import MeldCohort,MeldSubject
import sklearn.metrics as metrics
from meld_graph.evaluation import load_prediction, sens_spec_curves, roc_curves, plot_roc_multiple
import pandas as pd
import ptitprince as pt
#script to ensemble predictions on test set of multiple models.
#calculates bootstrapped predictions for statistical comparisons
#saves these out as data tables



model_paths = {'nnunet':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-22_DHAM_nnunet/s_0',
               'full_model:class+dist+finetuning':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_BAMH_classification_distance_finetuning/s_2',
               '-finetuning':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_WRZI_classification_distance/s_0',
               '-classification':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-23_HKOJ_finetuning_distance/s_2',
               '-distance':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-24_OSLA_finetuning_classification/s_2',
  

}
model_paths = {'nnunet':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-22_DHAM_nnunet/s_0',
               'classification':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-23_QUCI_classification/s_0',
               'distance':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_WRZI_distance/s_0',
               'distance+classification':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_WRZI_classification_distance/s_0',

}

model_paths = {'nnunet':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-06_FKKY_nnunet/s_0',
               'distance':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-06_FKKY_distance/s_0',
               'classification':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-23_QUCI_classification/s_0',
               'distance+classification':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_WRZI_classification_distance/s_0',
               'low_smooth':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-31_WQVY_raw/s_0',
               'low_smooth_regress':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-04-18_WLMU_thick_regress/s_0',
               'low_smooth_regress_thick_wpct':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-04-26_EEWS_wpct_regress/s_0/',
}
model_paths = {'miccai':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_WRZI_classification_distance/s_0',
               'low_smooth_regress_thick_RC':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-06-05_RMRC_low_smooth_regress_RC/s_0',
               'no_combat':'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-06-16_YRWZ_no_combat/s_0'

}


cohort = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat_6.hdf5',
               dataset='MELD_dataset_V6.csv')




save_dirs = {}
for model in model_paths.keys():
    save_dirs[model] = [os.path.join(model_paths[model],f'fold_0{fold}', 'results') for fold in np.arange(5)]

n_vert = len(cohort.cortex_label)*2
with h5py.File(os.path.join(save_dirs[model][0], 'predictions.hdf5'), "r") as f:
    subjects = list(f.keys())


#number of tresholds to evaluate ROC curve, between 0 & 1.
n_thresh=101
roc_curves_thresholds=np.linspace(0,1,n_thresh)
roc_dictionary_bs={}
roc_dictionary={}
for model_name in save_dirs.keys():

    roc_dictionary[model_name]={'sensitivity':np.zeros(n_thresh),
'sensitivity_plus':np.zeros(n_thresh),
'specificity':np.zeros(n_thresh)}
    roc_dictionary_bs[model_name]={}
    for fold in np.arange(5):
        roc_dictionary_bs[model_name][f'fold_0{fold}_bs']={'sensitivity':np.zeros(n_thresh),
'sensitivity_plus':np.zeros(n_thresh),
'specificity':np.ones(n_thresh)}
        roc_dictionary_bs[model_name][f'fold_0{fold}']={'sensitivity':np.zeros(n_thresh),
'sensitivity_plus':np.zeros(n_thresh),
'specificity':np.ones(n_thresh)}
        
    for si,subj in enumerate(subjects):
        if si%100==0:
            print(si)
        s = MeldSubject(subj,cohort=cohort)
        labels_hemis = {}
        dists={}
        subject_results = np.zeros((5,n_vert))
        labels = np.zeros(n_vert)
        for hemi in ['lh','rh']:
            dists[hemi], labels_hemis[hemi] = s.load_feature_lesion_data(
                        features=['.on_lh.boundary_zone.mgh'], hemi=hemi, features_to_ignore=[]
                    )
            if np.sum(dists[hemi])==0:
                dists[hemi] +=200
        labels = np.hstack([labels_hemis['lh'][cohort.cortex_mask],labels_hemis['rh'][cohort.cortex_mask]])
        borderzones = np.vstack([dists['lh'][cohort.cortex_mask,:],dists['rh'][cohort.cortex_mask,:]]).ravel()<20
        n_folds = len(save_dirs[model_name])
        n_folds = len(save_dirs[model_name])
        for fold in np.arange(n_folds):
            save_dir = save_dirs[model_name][fold]
            pred_file = os.path.join(save_dir, 'predictions.hdf5')
            result_hemis = load_prediction(subj,pred_file, dset='prediction')
            subject_results[fold] = np.hstack([result_hemis['lh'],result_hemis['rh']])
        m_subject_results = np.mean(subject_results,axis=0)
        subject_dictionary={'input_labels':labels,'borderzone':borderzones,'result':m_subject_results}
        roc_curves(subject_dictionary,roc_dictionary[model_name],roc_curves_thresholds)
        for fold in np.arange(n_folds):
            inds = np.random.choice(5,5)
            m_subject_results = np.mean(subject_results[inds],axis=0)
            subject_dictionary={'input_labels':labels,'borderzone':borderzones,'result':m_subject_results}
            roc_curves(subject_dictionary,roc_dictionary_bs[model_name][f'fold_0{fold}_bs'],roc_curves_thresholds)
            subject_dictionary={'input_labels':labels,'borderzone':borderzones,'result':subject_results[fold]}
            roc_curves(subject_dictionary,roc_dictionary_bs[model_name][f'fold_0{fold}'],roc_curves_thresholds)
    


model_path3 = '/rds/project/kw350/rds-kw350-meld/experiments/co-ripa1/iteration_21-09-15_nothresh/ensemble_21-09-15'
file= os.path.join(model_path3,f'fold_all', 'results', 'predictions_ensemble_iteration_0.hdf5')
roc_dictionary['per vertex']={'sensitivity':np.zeros(n_thresh),
'sensitivity_plus':np.zeros(n_thresh),
'specificity':np.zeros(n_thresh)}
for si,subj in enumerate(subjects):
    if si%100==0:
        print(si)
    s = MeldSubject(subj,cohort=cohort)
    labels_hemis = {}
    dists={}
    subject_results = np.zeros(n_vert)
    labels = np.zeros(n_vert)
    for hemi in ['lh','rh']:
        dists[hemi], labels_hemis[hemi] = s.load_feature_lesion_data(
                    features=['.on_lh.boundary_zone.mgh'], hemi=hemi, features_to_ignore=[]
                )
        if np.sum(dists[hemi])==0:
            dists[hemi] +=200
    labels = np.hstack([labels_hemis['lh'][cohort.cortex_mask],labels_hemis['rh'][cohort.cortex_mask]])
    borderzones = np.vstack([dists['lh'][cohort.cortex_mask,:],dists['rh'][cohort.cortex_mask,:]]).ravel()<20
    result_hemis = load_prediction(subj,file, dset='prediction_raw')
    subject_results =np.hstack([result_hemis['lh'],result_hemis['rh']])
    subject_dictionary={'input_labels':labels,'borderzone':borderzones,'result':subject_results}
    roc_curves(subject_dictionary,roc_dictionary['per vertex'],roc_curves_thresholds)


fig = plot_roc_multiple(roc_dictionary,roc_curves_thresholds)
fig.savefig('../figures/rocs.png')


df = []
for model in roc_dictionary.keys():
    sensitivity_curve,specificity_curve= sens_spec_curves(roc_dictionary[model])
    #this fixed value is what we used for iec.
    auc = metrics.auc(1-specificity_curve,sensitivity_curve)
    optimal_thresh = np.argmin(np.abs(sensitivity_curve-0.67))
    df.append([model,auc,sensitivity_curve[optimal_thresh],specificity_curve[optimal_thresh],roc_curves_thresholds[optimal_thresh]])
df = pd.DataFrame(df,columns=['Model',"AUC",'Sensitivity','Specificity','Threshold'])


df.to_csv('../data/test_stats.csv')

vertex_auc =0.64
df= []
df2 = []
for model_name in save_dirs.keys():
    for fold in np.arange(5):
        sensitivity_curve,specificity_curve= sens_spec_curves(roc_dictionary_bs[model_name][f'fold_0{fold}_bs'],
        )
        auc = metrics.auc(1-specificity_curve,sensitivity_curve)
        df.append([model_name,auc])
        sensitivity_curve,specificity_curve= sens_spec_curves(roc_dictionary_bs[model_name][f'fold_0{fold}'],
        )
        auc = metrics.auc(1-specificity_curve,sensitivity_curve)
        df2.append([model_name,auc])

df = pd.DataFrame(df,columns = ['Model','AUC'])
df['AUC'] = df['AUC'].astype(float)
df.to_csv('../data/bootstrapped_aucs.csv')

fig, ax = plt.subplots(1,1,figsize=(10,5))
pt.RainCloud(data=df, x='Model',y='AUC',ax=ax)
ax.plot([-0.5,len(list(save_dirs.keys()))],[vertex_auc,vertex_auc])
fig.savefig('../figures/bootstrapped_aucs.png')


df2= pd.DataFrame(df2,columns = ['Model','AUC'])
df2['AUC'] = df2['AUC'].astype(float)
df2.to_csv('../data/single_fold_aucs.csv')
fig, ax = plt.subplots(1,1,figsize=(10,5))
pt.RainCloud(data=df2, x='Model',y='AUC',ax=ax)
ax.plot([-0.5,len(list(save_dirs.keys()))],[vertex_auc,vertex_auc])
fig.savefig('../figures/single_fold_aucs.png')
