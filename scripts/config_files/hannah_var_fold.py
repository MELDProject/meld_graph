import random, string, datetime
import numpy as np
def date_code():
    # return unique date code of form: YYYY-MM-DD_XXXX
    return datetime.datetime.now().strftime("%y-%m-%d") + '_' + \
        ''.join(random.choices(string.ascii_uppercase, k=4))
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
    # large synthetic lesions
    {
        'network_parameters': {
            'training_parameters':{'num_epochs': 100,
                    'oversampling': False,
            }
        },
        'data_parameters': {
            'synthetic_data': {'run_synthetic': True, 'radius': 2},
        }
    },
    # small synthetic lesions
     {
        'network_parameters': {
            'training_parameters':{'num_epochs': 40,
                    'oversampling': False,
            }
        },
        'data_parameters': {
            'synthetic_data': {'run_synthetic': True, 'radius': 0.5},
        }
    },
    # real data
    {
        'network_parameters': {
            'training_parameters':{'num_epochs': 1000,
            'oversampling': True},
        },
        'data_parameters': {
            'synthetic_data': {'run_synthetic': False},
            'group':'both',
        }
    }
]

# parallel: these experiments are run in parallel. For each parallel experiment, all experiments in sequential will be launched=
parallel = []
losses=[
    {
        'network_parameters': {
            'name': base_name + '_baseline_dcd',
        },
        'data_parameters': {},
    },
    #{
    #    'network_parameters': {
    #        'name': base_name + '_baseline_nnunet',
    #        'training_parameters': {
    #            "loss_dictionary": {
    #                "cross_entropy": { "weight": 1 },
    #                "dice": { "class_weights": [0.0,1.0], "weight": 1},
    #                "distance_regression": {"loss": "mae", "weigh_by_gt": True, "weight": 0},
    #            },
    #        },
    #    },
    #    'data_parameters': {}
    #},
    {
        'network_parameters': {
            'name': base_name + '_large_augmentation',
        },
       'data_parameters': {
            "augment_data": {
                "augment_lesion": {"p": 0.0},
                "blur": {"p": 0.8},
                "brightness": {"p": 0.8},
                "contrast": {"p": 0.8},
                "extend_lesion": {"p": 0.0},
                "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
                "gamma": {"p": 0.15},
                "low_res": {"p": 0.8},
                "noise": {"p": 0.8},
                "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.8},
                "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.8},
            }
        }
    },
    #{
    #    'network_parameters': {
    #        'name': base_name + '_no_augmentation',
    #    },
    #    'data_parameters': {
    #        "augment_data": {
    #            "augment_lesion": {"p": 0.0},
    #            "blur": {"p": 0.0},
    #            "brightness": {"p": 0.0},
    #            "contrast": {"p": 0.0},
    #            "extend_lesion": {"p": 0.0},
    #            "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.0},
    #            "gamma": {"p": 0.0},
    #            "low_res": {"p": 0.0},
    #            "noise": {"p": 0.0},
    #            "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.0},
    #            "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.0},
    #        }
    #    }
    #},
    #{
    #    # NOTE must be created using hannah_based_smooth_lesion
    #    'network_parameters': {
    #        'name': base_name + '_smooth_lesion',
    #    },
    #    'data_parameters': {
    #        "smooth_labels": True,
    #    }
    #},
    {
        'network_parameters': {
            'name': base_name + '_mask_augmentation',
        },
        'data_parameters': {
            "augment_data": {
                "augment_lesion": {"p": 0.2},
            }
        }
    }
    ]

from copy import deepcopy
for loss in losses:
    for fold in np.arange(5):
        loss['data_parameters']['fold_n'] = fold
        parallel.append(deepcopy(loss))


