import meld_graph
import meld_graph.models
import meld_graph.experiment
import meld_graph.dataset

import logging
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
    parser = argparse.ArgumentParser(description="""
        Train model using config in config_file
        """)
    parser.add_argument("--config_file", help="path to experiment_config.py", default="config_files/experiment_config.py")
    parser.add_argument("--wandb_logging", action="store_true", help="enable wandb logging.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    
    # create experiment
    exp = meld_graph.experiment.Experiment(config.network_parameters, config.data_parameters, verbose=logging.INFO)
    # train the model
    exp.train()