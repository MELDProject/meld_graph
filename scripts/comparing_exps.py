import numpy as np

# from scipy.spatial.distance import mahalanobis
import scipy.linalg as sp
from meld_classifier.meld_cohort import MeldCohort
from meld_graph.data_preprocessing import Preprocess as Prep
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from pygam import LinearGAM

exps = {
    "baseline": {
        "hdf5_file_root": "{site_code}_{group}_featurematrix_combat_6_kernels.hdf5",
        "features": [
            ".combat.on_lh.sulc.sm3.mgh",
            ".combat.on_lh.curv.sm3.mgh",
            ".combat.on_lh.pial.K_filtered.sm20.mgh",
            ".combat.on_lh.thickness.sm3.mgh",
            ".combat.on_lh.w-g.pct.sm3.mgh",
            ".combat.on_lh.thickness_regression.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_1.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
            ".inter_z.intra_z.combat.on_lh.thickness.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.thickness.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ],
        "preprocessing_parameters": {
            "scaling": None,
            "zscore": "../data/feature_means_kernel3.json",
        },
    },
    "msm": {
        "hdf5_file_root": "{site_code}_{group}_featurematrix_combat_msm.hdf5",
        "features": [
            ".combat.on_lh.sulc.sm3.mgh",
            ".combat.on_lh.curv.sm3.mgh",
            ".combat.on_lh.pial.K_filtered.sm20.mgh",
            ".combat.on_lh.thickness.sm3.mgh",
            ".combat.on_lh.w-g.pct.sm3.mgh",
            ".combat.on_lh.thickness_regression.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".combat.on_lh.wm_FLAIR_1.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
            ".inter_z.intra_z.combat.on_lh.thickness.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.thickness.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
            ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ],
        "preprocessing_parameters": {
            "scaling": None,
            "zscore": "../data/feature_means_msm.json",
        },
    },
}

data_parameters = {
    "augment_data": {},
    "combine_hemis": None,
    "dataset": "MELD_dataset_V6.csv",
    "hdf5_file_root": "{site_code}_{group}_featurematrix_combat_6.hdf5",
    "features": [
        ".inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.intra_z.combat.on_lh.thickness.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.w-g.pct.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh",
        ".inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.thickness.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh",
    ],
    "features_to_exclude": [],
    "features_to_replace_with_0": [],
    "fold_n": 0,
    "group": "control",
    "icosphere_parameters": {"distance_type": "exact"},
    "lesion_bias": 0,
    "lobes": False,
    "number_of_folds": 5,
    "preprocessing_parameters": {
        "scaling": None,
        "zscore": "../data/feature_means.json",
    },
    "scanners": ["15T", "3T"],
    "site_codes": [
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
    ],
    "smooth_labels": False,
    "subject_features_to_exclude": [],
    "synthetic_data": {
        "bias": 1,
        "jitter_factor": 2,
        "n_subs": 500,
        "n_subtypes": 25,
        "proportion_features_abnormal": 1,
        "proportion_hemispheres_lesional": 0.5,
        "radius": 0.5,
        "run_synthetic": False,
        "smooth_lesion": False,
        "use_controls": True,
    },
}


def patient_data(patient_ids, prep):
    """returns lesions, flair features and non-flair_features"""
    lesions = []
    non_flair_features = []
    flair_features = []
    for pi, patient in enumerate(patient_ids):
        if pi % 100 == 0:
            print(pi)
        left, right = prep.get_data_preprocessed(
            subject=patient,
            features=data_parameters["features"],
            lobes=data_parameters["lobes"],
            lesion_bias=False,
        )
        if left["labels"].sum() > 0:
            data = left
        elif right["labels"].sum() > 0:
            data = right
        lesion = data["labels"]
        lesions.append(lesion)
        # lesional_features_means = np.mean(data['features'][lesion==1],axis=0)
        inds = np.random.choice(np.where(lesion == 1)[0], 100)
        lesional_features = data["features"][inds]
        flair_mask = np.zeros(len(data_parameters["features"]), dtype=bool)
        for fi, f in enumerate(data_parameters["features"]):
            if "FLAIR" in f:
                flair_mask[fi] = True
        non_flair = lesional_features[:, ~flair_mask]
        flair = lesional_features[:, flair_mask]
        non_flair_features.append(non_flair)
        if flair.sum() != 0:
            flair_features.append(flair)

    non_flair_features_stacked = np.vstack(non_flair_features)
    flair_features_stacked = np.vstack(flair_features)
    return non_flair_features_stacked, flair_features_stacked, lesions


def control_data(control_ids, prep):
    # load in control data
    control_non_flair_features = []
    control_flair_features = []
    for ci, control in enumerate(control_ids):
        if ci % 100 == 0:
            print(ci)
        left, right = prep.get_data_preprocessed(
            subject=control,
            features=data_parameters["features"],
            lobes=data_parameters["lobes"],
            lesion_bias=False,
        )
        data = np.random.choice([left, right])
        inds = np.random.choice(cohort.cortex_label, 100)
        #        lesional_features_means = np.mean(data['features'][lesion==1],axis=0)
        lesional_features = data["features"][inds]
        flair_mask = np.zeros(len(data_parameters["features"]), dtype=bool)
        for fi, f in enumerate(data_parameters["features"]):
            if "FLAIR" in f:
                flair_mask[fi] = True

        non_flair = lesional_features[:, ~flair_mask]
        flair = lesional_features[:, flair_mask]
        control_non_flair_features.append(non_flair)
        if flair.sum() != 0:
            control_flair_features.append(flair)

    control_non_flair_features_stacked = np.vstack(control_non_flair_features)
    control_flair_features_stacked = np.vstack(control_flair_features)
    return control_non_flair_features_stacked, control_flair_features_stacked


save_features = {}
for exp in exps.keys():
    save_features[exp] = {}
    var_params = exps[exp]
    for param in var_params.keys():
        data_parameters[param] = var_params[param]

    cohort = MeldCohort(
        hdf5_file_root=data_parameters["hdf5_file_root"],
        dataset=data_parameters["dataset"],
    )
    controls = cohort.get_subject_ids(group="control")
    patients = cohort.get_subject_ids(group="patient")
    # restrict to trainval cohort
    _, trainval, test = cohort.read_subject_ids_from_dataset()
    train_patients = np.intersect1d(trainval, patients)
    train_controls = np.intersect1d(trainval, controls)
    prep = Prep(cohort=cohort, params=data_parameters)
    # update this for no FLAIR
    non_flair_features_stacked, flair_features_stacked, lesions = patient_data(train_patients, prep)
    control_non_flair_features_stacked, control_flair_features_stacked = control_data(train_controls, prep)
    save_features[exp]["p"] = non_flair_features_stacked
    save_features[exp]["pf"] = flair_features_stacked
    save_features[exp]["c"] = control_non_flair_features_stacked
    save_features[exp]["cf"] = control_flair_features_stacked


# organise into train and val
tf = ["", "f"]
n_folds = 5
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=[10, 5], random_state=1, max_iter=400)
res = {}
exp_names = list(exps.keys())
for exp in exps:
    res[exp] = {
        "s": [],
        "sf": [],
        "sy": [],
        "sfy": [],
    }

tf = ["", "f"]
for t in tf:
    for fold in np.arange(n_folds):
        for exp in exps:
            f = fold
            n_v = len(save_features[exp][f"p{t}"])
            n_c = len(save_features[exp][f"c{t}"])
            n_pats = n_v // 100
            n_conts = n_c // 100
            fold_step_p = 100 * n_pats // n_folds
            fold_step_c = 100 * n_conts // n_folds
            yc = np.zeros(n_c)
            yv = np.ones(n_v)
            val_x = np.vstack(
                [
                    save_features[exp][f"p{t}"][fold_step_p * f : fold_step_p * (f + 1)],
                    save_features[exp][f"c{t}"][fold_step_c * f : fold_step_c * (f + 1)],
                ]
            )
            val_y = np.concatenate(
                [
                    yv[fold_step_p * f : fold_step_p * (f + 1)],
                    yc[fold_step_c * f : fold_step_c * (f + 1)],
                ]
            )
            train_x = np.vstack(
                [
                    save_features[exp][f"p{t}"][: fold_step_p * f],
                    save_features[exp][f"p{t}"][fold_step_p * (f + 1) :],
                    save_features[exp][f"c{t}"][: fold_step_c * f],
                    save_features[exp][f"c{t}"][fold_step_c * (f + 1) :],
                ]
            )
            train_y = np.concatenate(
                [
                    yv[: fold_step_p * f],
                    yv[fold_step_p * (f + 1) :],
                    yc[: fold_step_c * f],
                    yc[fold_step_c * (f + 1) :],
                ]
            )
            clf.fit(train_x, train_y)
            pred_y = clf.predict_proba(val_x)
            res[exp][f"s{t}"].extend(pred_y[:, 1])
            res[exp][f"s{t}y"].extend(val_y)
            print(exp, np.round(roc_auc_score(val_y, pred_y[:, 1]), 2))

t = "f"
results = []
for exp in exps:
    auc = roc_auc_score(res[exp][f"s{t}y"], res[exp][f"s{t}"])
    print(exp, auc)
    results.append([exp, auc])

t = ""
for exp in exps:
    auc = roc_auc_score(res[exp][f"s{t}y"], res[exp][f"s{t}"])
    print(exp, auc)
    results.append([exp, auc])
