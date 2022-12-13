import random, string, datetime

def date_code():
    # return unique date code of form: YYYY-MM-DD_XXXX
    return datetime.datetime.now().strftime("%y-%m-%d") + '_' + \
        ''.join(random.choices(string.ascii_uppercase, k=4))
# set up multiple configs at the same time

# base name for all experiments
base_name = date_code() + '_test'

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
            'training_parameters':{'num_epochs': 50},
        },
        'data_parameters': {
            'synthetic_data': {'run_synthetic': True, 'radius': 2},
        }
    },
    # small synthetic lesions
     {
        'network_parameters': {
            'training_parameters':{'num_epochs': 50},
        },
        'data_parameters': {
            'synthetic_data': {'run_synthetic': True, 'radius': 0.5},
        }
    },
    # real data
    {
        'network_parameters': {
            'training_parameters':{'num_epochs': 1000},
        },
        'data_parameters': {
            'synthetic_data': {'run_synthetic': False},
        }
    }
]

# parallel: these experiments are run in parallel. For each parallel experiment, all experiments in sequential will be launched=
parallel = [
{'data_parameters': {'fold_n': 0},
  'network_parameters': {'name': base_name}},
{'data_parameters': {'fold_n': 1},
'network_parameters': {'name': base_name}},
{'data_parameters': {'fold_n': 2},
'network_parameters': {'name': base_name}},
{'data_parameters': {'fold_n': 3},
'network_parameters': {'name': base_name}},
{'data_parameters': {'fold_n': 4},
'network_parameters': {'name': base_name}},
]