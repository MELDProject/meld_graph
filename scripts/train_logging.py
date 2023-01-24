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

def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config

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
    
    # create experiment
    exp = meld_graph.experiment.Experiment(config.network_parameters, config.data_parameters, verbose=logging.INFO)
    exp.load_model()
    #initiate logging
    # train the model
    exp.train(wandb_logging=True)
