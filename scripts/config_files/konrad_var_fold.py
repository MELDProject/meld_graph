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
            'training_parameters':{'num_epochs': 50,
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
            'training_parameters':{'num_epochs': 20,
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
losses=[{'network_parameters': {
    'name': base_name + '_additional_mae_loss',

    'training_parameters':{
        'loss_dictionary': {  
            #'cross_entropy':{'weight':1},
           # 'focal_loss':{'weight':1, 'alpha':0.4, 'gamma':4},
         #   'dice':{'weight': 1, 'class_weights': [.0, 1.0]},
            'mae_loss':{'weight':1},
            'distance_regression': {'weight': 1, 'weigh_by_gt': True,
            'loss':'mae'}
            }
        }
    }
    },
    
        ]

from copy import deepcopy
for loss in losses:
    for fold in np.arange(10):
        loss['data_parameters']={'fold_n': fold}
        parallel.append(deepcopy(loss))


