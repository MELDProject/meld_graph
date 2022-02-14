
#import importlib
import meld_graph
import meld_graph.models
import meld_graph.experiment
import meld_graph.dataset
import numpy as np
#importlib.reload(meld_graph)
#importlib.reload(meld_graph.models)
#importlib.reload(meld_graph.dataset)
#importlib.reload(meld_graph.experiment)

import logging
logging.basicConfig(level=logging.INFO)
import argparse

def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to experiment_config.py", default="experiment_config.py")
    args = parser.parse_args()

    config = load_config(args.config_file)
    # create experiment
    exp = meld_graph.experiment.Experiment(config.network_parameters, config.data_parameters, save=False)
    # TODO: manual selection of train/val ids for testing
    #_ = exp.get_train_val_test_ids()
    #exp.data_parameters['train_ids'] = ['MELD_H4_3T_FCD_0011'] #, 'MELD_H4_3T_FCD_0011'] #exp.data_parameters['train_ids'][:10]
    #exp.data_parameters['val_ids'] = ['MELD_H4_3T_FCD_0011'] #, 'MELD_H4_3T_FCD_0011'] #exp.data_parameters['train_ids'][:10]
    # train the model
    exp.train()