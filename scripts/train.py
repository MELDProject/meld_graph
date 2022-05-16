
#import importlib
import meld_graph
import meld_graph.models
import meld_graph.experiment
import meld_graph.dataset
from meld_graph.paths import EXPERIMENT_PATH
import numpy as np

import logging
import argparse
from copy import deepcopy
import os
from functools import reduce
import operator
from train import load_config
import datetime

def nested_get(d, keys):
    return reduce(operator.getitem, keys, d)

def nested_set(d, keys, value):
    nested_get(d, keys[:-1])[keys[-1]] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Train model with variable network or data parameters. 
        If config contains variable_parameters dict, will iterate over all params in this dict and create experiments.
        """)
    parser.add_argument("--config_file", help="path to experiment_config.py", default="config_files/experiment_config.py")
    args = parser.parse_args()

    config = load_config(args.config_file)
    variable_parameters = getattr(config, 'variable_parameters', {})
    # create and run experiments
    for param, values in config.variable_parameters.items():
        params = param.split('$')
        path = '$'.join(params[1:])
        for value in values:
            name = f'{path}_{value}'
            print(f'Starting experiment {path}, {name}')
            cur_data_parameters = deepcopy(config.data_parameters)
            cur_network_parameters = deepcopy(config.network_parameters)

            # set name
            cur_network_parameters['name'] = f'{datetime.datetime.now().strftime("%y-%m-%d")}_{path}/{name}'
            # change variable params
            if params[0] == 'network_parameters':
                nested_set(cur_network_parameters, params[1:], value)
            elif params[0] == 'data_parameters':
                nested_set(cur_data_parameters, params[1:], value)
                pass
            else:
                NotImplementedError(params[0])

            # create experiment
            exp = meld_graph.experiment.Experiment(cur_network_parameters, cur_data_parameters, verbose=logging.INFO)
            # train the model
            exp.train()
    if len(variable_parameters) == 0:
        # only one experiment to train
        # create experiment
        exp = meld_graph.experiment.Experiment(config.network_parameters, config.data_parameters, verbose=logging.INFO)
        # train the model
        exp.train()