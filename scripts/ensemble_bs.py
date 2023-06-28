import meld_graph.experiment
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import nibabel as nb
from meld_classifier.paths import BASE_PATH
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import pandas as pd
import ptitprince as pt


def optimal_threshold(b, roc_curves_thresholds):
    sensitivity_curve = b["sensitivity_plus"] / max(b["sensitivity_plus"])
    specificity_curve = b["specificity"] / max(b["specificity"])
    youden = sensitivity_curve + specificity_curve
    optimal_thresh = np.max(np.where(youden == np.max(youden)))
    return sensitivity_curve, specificity_curve, optimal_thresh


import sklearn.metrics as metrics


def plot_roc(sensitivity_curve, specificity_curve, optimal_thresh, roc_curves_thresholds):
    fig, axes = plt.subplots(1, 2)
    axes = axes.ravel()
    axes[0].plot(
        1 - specificity_curve,
        sensitivity_curve,
    )
    # axes[0].scatter(1-specificity_curve[optimal_thresh],sensitivity_curve[optimal_thresh], c='r')
    axes[1].plot(roc_curves_thresholds, sensitivity_curve, label="sensitivity")
    axes[1].plot(roc_curves_thresholds, specificity_curve, label="specificity")
    axes[1].plot(roc_curves_thresholds, specificity_curve + sensitivity_curve - 1, label="Youden")
    # axes[1].plot([roc_curves_thresholds[optimal_thresh],roc_curves_thresholds[optimal_thresh]],
    # [0,1],label='Optimal threshold')
    fig.legend()
    auc = metrics.auc(1 - specificity_curve, sensitivity_curve)
    return fig


def plot_roc_multiple(roc_dictionary, roc_curves_thresholds):
    fig, ax = plt.subplots(1, 1)
    for mi, model in enumerate(roc_dictionary.keys()):
        sensitivity_curve, specificity_curve, optimal_thresh = optimal_threshold(
            roc_dictionary[model], roc_curves_thresholds=roc_curves_thresholds
        )
        ax.plot(1 - specificity_curve, sensitivity_curve, label=model)
        auc = metrics.auc(1 - specificity_curve, sensitivity_curve)
        ax.text(0, 1 - mi / 10, f"{model} AUC: {auc:.2f}")
    fig.legend()
    return fig


def load_prediction(subject, hdf5, dset="prediction"):
    results = {}
    with h5py.File(hdf5, "r") as f:
        for hemi in ["lh", "rh"]:
            results[hemi] = f[subject][hemi][dset][:]
    return results


def roc_curves(subject_dictionary, roc_dictionary, roc_curves_thresholds):
    """calculate performance at multiple thresholds"""

    for t_i, threshold in enumerate(roc_curves_thresholds):
        predicted = subject_dictionary["result"] >= threshold
        # if we want tpr vs fpr curve too
        # tp,fp,fn, tn = tp_fp_fn_tn(predicted, subject_dictionary['input_labels'])
        # store sensitivity and sensitivity_plus for each patient (has a label)
        if subject_dictionary["input_labels"].sum() > 0:
            roc_dictionary["sensitivity"][t_i] += np.logical_and(predicted, subject_dictionary["input_labels"]).any()
            bordered = np.logical_and(predicted, subject_dictionary["borderzone"]).any()
            roc_dictionary["sensitivity_plus"][t_i] += bordered
            if not bordered:
                break
        # store specificity for controls (no label)
        else:
            no_fps = ~predicted.any()
            roc_dictionary["specificity"][t_i:] += no_fps
            if no_fps:
                break


model_paths = {
    "nnunet": "/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-06_FKKY_nnunet/s_0",
    "distance": "/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-06_FKKY_distance/s_0",
    "classification": "/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-02-23_QUCI_classification/s_0",
    "distance+classification": "/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-01_WRZI_classification_distance/s_0",
    "raw": "/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-03-31_WQVY_raw/s_0",
}

save_dirs = {}
for model in model_paths.keys():
    save_dirs[model] = [os.path.join(model_paths[model], f"fold_0{fold}", "results") for fold in np.arange(5)]

cohort = MeldCohort(
    hdf5_file_root="{site_code}_{group}_featurematrix_combat_6.hdf5",
    dataset="MELD_dataset_V6.csv",
)


n_vert = len(cohort.cortex_label) * 2
with h5py.File(os.path.join(save_dirs["nnunet"][0], "predictions.hdf5"), "r") as f:
    subjects = list(f.keys())

# number of tresholds to evaluate ROC curve, between 0 & 1.
n_thresh = 101
roc_curves_thresholds = np.linspace(0, 1, n_thresh)
roc_dictionary_base = {
    "sensitivity": np.zeros(n_thresh),
    "sensitivity_plus": np.zeros(n_thresh),
    "specificity": np.zeros(n_thresh),
}
# bootstrapping
roc_dictionary_bs = {}
for model_name in save_dirs.keys():
    roc_dictionary_bs[model_name] = {}
    for fold in np.arange(5):
        roc_dictionary_bs[model_name][f"fold_0{fold}_bs"] = {
            "sensitivity": np.zeros(n_thresh),
            "sensitivity_plus": np.zeros(n_thresh),
            "specificity": np.ones(n_thresh),
        }
        roc_dictionary_bs[model_name][f"fold_0{fold}"] = {
            "sensitivity": np.zeros(n_thresh),
            "sensitivity_plus": np.zeros(n_thresh),
            "specificity": np.ones(n_thresh),
        }

    for si, subj in enumerate(subjects):
        if si % 100 == 0:
            print(si)
        s = MeldSubject(subj, cohort=cohort)
        labels_hemis = {}
        dists = {}
        subject_results = np.zeros((5, n_vert))
        labels = np.zeros(n_vert)
        for hemi in ["lh", "rh"]:
            dists[hemi], labels_hemis[hemi] = s.load_feature_lesion_data(
                features=[".on_lh.boundary_zone.mgh"], hemi=hemi, features_to_ignore=[]
            )
            if np.sum(dists[hemi]) == 0:
                dists[hemi] += 200
        labels = np.hstack(
            [
                labels_hemis["lh"][cohort.cortex_mask],
                labels_hemis["rh"][cohort.cortex_mask],
            ]
        )
        borderzones = np.vstack([dists["lh"][cohort.cortex_mask, :], dists["rh"][cohort.cortex_mask, :]]).ravel() < 20
        n_folds = len(save_dirs[model_name])
        for fold in np.arange(n_folds):
            save_dir = save_dirs[model_name][fold]
            pred_file = os.path.join(save_dir, "predictions.hdf5")
            result_hemis = load_prediction(subj, pred_file, dset="prediction")
            subject_results[fold] = np.hstack([result_hemis["lh"], result_hemis["rh"]])
        for fold in np.arange(n_folds):
            inds = np.random.choice(5, 5)
            m_subject_results = np.mean(subject_results[inds], axis=0)
            subject_dictionary = {
                "input_labels": labels,
                "borderzone": borderzones,
                "result": m_subject_results,
            }
            roc_curves(
                subject_dictionary,
                roc_dictionary_bs[model_name][f"fold_0{fold}_bs"],
                roc_curves_thresholds,
            )
            subject_dictionary = {
                "input_labels": labels,
                "borderzone": borderzones,
                "result": subject_results[fold],
            }
            roc_curves(
                subject_dictionary,
                roc_dictionary_bs[model_name][f"fold_0{fold}"],
                roc_curves_thresholds,
            )


vertex_auc = 0.64
df = []
df2 = []
for model_name in save_dirs.keys():
    for fold in np.arange(5):
        sensitivity_curve, specificity_curve, optimal_thresh = optimal_threshold(
            roc_dictionary_bs[model_name][f"fold_0{fold}_bs"],
            roc_curves_thresholds=roc_curves_thresholds,
        )
        auc = metrics.auc(1 - specificity_curve, sensitivity_curve)
        df.append([model_name, auc])
        sensitivity_curve, specificity_curve, optimal_thresh = optimal_threshold(
            roc_dictionary_bs[model_name][f"fold_0{fold}"],
            roc_curves_thresholds=roc_curves_thresholds,
        )
        auc = metrics.auc(1 - specificity_curve, sensitivity_curve)
        df2.append([model_name, auc])

df = pd.DataFrame(df, columns=["Model", "AUC"])
df["AUC"] = df["AUC"].astype(float)
df.to_csv("../data/bootstrapped_aucs.csv")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pt.RainCloud(data=df, x="Model", y="AUC", ax=ax)
ax.plot([-0.5, len(list(save_dirs.keys()))], [vertex_auc, vertex_auc])
fig.savefig("../figures/bootstrapped_aucs.png")


df2 = pd.DataFrame(df2, columns=["Model", "AUC"])
df2["AUC"] = df2["AUC"].astype(float)
df2.to_csv("../data/single_fold_aucs.csv")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pt.RainCloud(data=df2, x="Model", y="AUC", ax=ax)
ax.plot([-0.5, len(list(save_dirs.keys()))], [vertex_auc, vertex_auc])
fig.savefig("../figures/single_fold_aucs.png")
