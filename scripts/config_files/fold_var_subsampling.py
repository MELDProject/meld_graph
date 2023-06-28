# script to set up the subsampling experiments for MICCAI

import random, string, datetime
import numpy as np


def date_code():
    # return unique date code of form: YYYY-MM-DD_XXXX
    return datetime.datetime.now().strftime("%y-%m-%d") + "_" + "".join(random.choices(string.ascii_uppercase, k=4))


# set up multiple configs at the same time

# base name for all experiments
base_name = date_code()
# structure of experiment folder:
# parallel_name / s_X / fold_XX
# this means that for all experiments in parallels, sequential experiments are run

# sequential: these configs will be run in order, finetuning from the best model from the previous config
# should be a list of nested dicts defining parameters that change compared to the base config
# NOTE the finetuning flag is set automatically when generating the configs with create_config.py
sequential = [
    # real data
    {
        "network_parameters": {
            "training_parameters": {"num_epochs": 1000, "oversampling": True},
        },
        "data_parameters": {
            "synthetic_data": {"run_synthetic": False},
            "group": "both",
        },
    }
]

# parallel: these experiments are run in parallel. For each parallel experiment, all experiments in sequential will be launched=
parallel = []
# losses
losses = [
    {
        "network_parameters": {
            "name": base_name + "_subsampling_0.1",
            "training_parameters": {
                "deep_supervision": {
                    "levels": [6, 5, 4, 3, 2, 1],
                    "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
                },
                "loss_dictionary": {
                    "cross_entropy": {"weight": 1},
                    "dice": {"class_weights": [0.0, 1.0], "weight": 1},
                    "distance_regression": {
                        "loss": "mae",
                        "weigh_by_gt": True,
                        "weight": 1,
                    },
                    "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
                },
                "stopping_metric": {"name": "loss", "sign": 1},
                "metric_smoothing": False,
                "metrics": [
                    "dice_lesion",
                    "dice_nonlesion",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "auroc",
                    "cl_precision",
                    "cl_recall",
                ],
            },
        },
        "data_parameters": {
            "subsample_cohort_fraction": 0.1,
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.2},
                "brightness": {"p": 0.15},
                "contrast": {"p": 0.15},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.25},
                "noise": {"p": 0.15},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
            },
        },
    },
    # exp 2
    {
        "network_parameters": {
            "name": base_name + "_subsampling_0.2",
            "training_parameters": {
                "deep_supervision": {
                    "levels": [6, 5, 4, 3, 2, 1],
                    "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
                },
                "loss_dictionary": {
                    "cross_entropy": {"weight": 1},
                    "dice": {"class_weights": [0.0, 1.0], "weight": 1},
                    "distance_regression": {
                        "loss": "mae",
                        "weigh_by_gt": True,
                        "weight": 1,
                    },
                    "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
                },
                "stopping_metric": {"name": "loss", "sign": 1},
                "metric_smoothing": False,
                "metrics": [
                    "dice_lesion",
                    "dice_nonlesion",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "auroc",
                    "cl_precision",
                    "cl_recall",
                ],
            },
        },
        "data_parameters": {
            "subsample_cohort_fraction": 0.2,
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.2},
                "brightness": {"p": 0.15},
                "contrast": {"p": 0.15},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.25},
                "noise": {"p": 0.15},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
            },
        },
    },
    #
    {
        "network_parameters": {
            "name": base_name + "_subsampling_0.3",
            "training_parameters": {
                "deep_supervision": {
                    "levels": [6, 5, 4, 3, 2, 1],
                    "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
                },
                "loss_dictionary": {
                    "cross_entropy": {"weight": 1},
                    "dice": {"class_weights": [0.0, 1.0], "weight": 1},
                    "distance_regression": {
                        "loss": "mae",
                        "weigh_by_gt": True,
                        "weight": 1,
                    },
                    "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
                },
                "stopping_metric": {"name": "loss", "sign": 1},
                "metric_smoothing": False,
                "metrics": [
                    "dice_lesion",
                    "dice_nonlesion",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "auroc",
                    "cl_precision",
                    "cl_recall",
                ],
            },
        },
        "data_parameters": {
            "subsample_cohort_fraction": 0.3,
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.2},
                "brightness": {"p": 0.15},
                "contrast": {"p": 0.15},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.25},
                "noise": {"p": 0.15},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
            },
        },
    },
    {
        "network_parameters": {
            "name": base_name + "_subsampling_0.4",
            "training_parameters": {
                "deep_supervision": {
                    "levels": [6, 5, 4, 3, 2, 1],
                    "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
                },
                "loss_dictionary": {
                    "cross_entropy": {"weight": 1},
                    "dice": {"class_weights": [0.0, 1.0], "weight": 1},
                    "distance_regression": {
                        "loss": "mae",
                        "weigh_by_gt": True,
                        "weight": 1,
                    },
                    "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
                },
                "stopping_metric": {"name": "loss", "sign": 1},
                "metric_smoothing": False,
                "metrics": [
                    "dice_lesion",
                    "dice_nonlesion",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "auroc",
                    "cl_precision",
                    "cl_recall",
                ],
            },
        },
        "data_parameters": {
            "subsample_cohort_fraction": 0.4,
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.2},
                "brightness": {"p": 0.15},
                "contrast": {"p": 0.15},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.25},
                "noise": {"p": 0.15},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
            },
        },
    },
    #
    {
        "network_parameters": {
            "name": base_name + "_subsampling_0.6",
            "training_parameters": {
                "deep_supervision": {
                    "levels": [6, 5, 4, 3, 2, 1],
                    "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
                },
                "loss_dictionary": {
                    "cross_entropy": {"weight": 1},
                    "dice": {"class_weights": [0.0, 1.0], "weight": 1},
                    "distance_regression": {
                        "loss": "mae",
                        "weigh_by_gt": True,
                        "weight": 1,
                    },
                    "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
                },
                "stopping_metric": {"name": "loss", "sign": 1},
                "metric_smoothing": False,
                "metrics": [
                    "dice_lesion",
                    "dice_nonlesion",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "auroc",
                    "cl_precision",
                    "cl_recall",
                ],
            },
        },
        "data_parameters": {
            "subsample_cohort_fraction": 0.6,
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.2},
                "brightness": {"p": 0.15},
                "contrast": {"p": 0.15},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.25},
                "noise": {"p": 0.15},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
            },
        },
    },
    {
        "network_parameters": {
            "name": base_name + "_subsampling_0.8",
            "training_parameters": {
                "deep_supervision": {
                    "levels": [6, 5, 4, 3, 2, 1],
                    "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
                },
                "loss_dictionary": {
                    "cross_entropy": {"weight": 1},
                    "dice": {"class_weights": [0.0, 1.0], "weight": 1},
                    "distance_regression": {
                        "loss": "mae",
                        "weigh_by_gt": True,
                        "weight": 1,
                    },
                    "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
                },
                "stopping_metric": {"name": "loss", "sign": 1},
                "metric_smoothing": False,
                "metrics": [
                    "dice_lesion",
                    "dice_nonlesion",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "auroc",
                    "cl_precision",
                    "cl_recall",
                ],
            },
        },
        "data_parameters": {
            "subsample_cohort_fraction": 0.8,
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.2},
                "brightness": {"p": 0.15},
                "contrast": {"p": 0.15},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.25},
                "noise": {"p": 0.15},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
            },
        },
    },
]


from copy import deepcopy

for loss in losses:
<<<<<<< HEAD
    for fold in np.arange(5):
        if 'data_parameters' in loss.keys():
            loss['data_parameters']['fold_n'] =fold
=======
    for fold in np.arange(1):
        if "data_parameters" in loss.keys():
            loss["data_parameters"]["fold_n"] = fold
>>>>>>> de2692f1a214d48b8bacdf071f922206f1229de1
        else:
            loss["data_parameters"] = {"fold_n": fold}
        parallel.append(deepcopy(loss))
