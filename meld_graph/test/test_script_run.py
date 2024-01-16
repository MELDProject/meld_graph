#### tests for scripts/run.py ####
# this is a long test, if want to exclude it for quick testing, run
# pytest -m "not slow"
# tested functionality:
#   load_config, save_config
#   overall training / evaluation functionality
#   creation of expected folders when iterating over network or data parameters
#   evaluation
# MISSING TESTS
# - Model comparison
# - prediction of individual subjects

import datetime
import subprocess
import tempfile
import os
import glob
from meld_graph.experiment import Experiment, load_config, save_config
import pytest
from meld_graph.paths import DEFAULT_HDF5_FILE_ROOT


def get_data_parameters():
    data_parameters = {
        "site_codes": ["TEST"],
        "scanners": ["15T", "3T"],
        "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
        "dataset": "MELD_dataset_TEST.csv",
        "group": "both",
        "features_to_exclude": [],
        "subject_features_to_exclude": [""],
        "min_area_threshold": 50,
        "number_of_folds": 10,
        "fold_n": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "iteration": 0,
        "batch_size": 1024,
        "contra": False,
        "num_per_subject": None,
        "equalize": True,
        "equalize_factor": 1,
        "active_selection": True,
        "active_selection_pool_factor": 5,
        "active_selection_frac": 0.5,
        "resample_each_epoch": False,
        "shuffle_each_epoch": True,
        "universal_features": "",
        "num_neighbours": 0,
    }
    return data_parameters


def get_network_parameters():
    network_parameters = {
        ##### network architecture #####
        "layer_sizes": [20],
        "dropout": 0.2,
        ##### training hyper-params #####
        "learning_rate": 0.0001,
        "max_patience": 10,
        "num_epochs": 1,
        "loss": "binary_crossentropy",
        "weighting": None,
        "optimal_threshold": 0.4,
        "date": datetime.datetime.now().strftime("%y-%m-%d"),
    }
    return network_parameters


def create_config_file(
    fname, data_parameters, network_parameters, variable_data_parameters, variable_network_parameters
):
    with open(fname, "w") as f:
        f.write("variable_data_parameters = ")
        f.write(repr(variable_data_parameters))
        f.write("\n")
        f.write("variable_network_parameters = ")
        f.write(repr(variable_network_parameters))
        f.write("\n")
        f.write("data_parameters = ")
        f.write(repr(data_parameters))
        f.write("\n")
        f.write("network_parameters = ")
        f.write(repr(network_parameters))
        f.write("\n")


def test_save_load_config_file():
    # get data parameters
    data_parameters = get_data_parameters()
    data_parameters["number_of_folds"] = 10
    data_parameters["fold_n"] = [0, 9]
    variable_data_parameters = {"iteration": [0, 1]}
    network_parameters = get_network_parameters()
    variable_network_parameters = {"learning_rate": [0.001, 0.005]}
    print(network_parameters)
    # create config file
    with tempfile.NamedTemporaryFile(suffix=".py") as config_file:
        create_config_file(
            fname=config_file.name,
            data_parameters=data_parameters,
            network_parameters=network_parameters,
            variable_data_parameters=variable_data_parameters,
            variable_network_parameters=variable_network_parameters,
        )
        # load that config file
        config = load_config(config_file.name)
        # check that all parameters are the same that we have written
        assert config.variable_data_parameters == variable_data_parameters
        assert config.variable_network_parameters == variable_network_parameters
        assert config.network_parameters == network_parameters
        assert config.data_parameters == data_parameters
        # save the config file
        with tempfile.NamedTemporaryFile(suffix=".py") as config_file2:
            save_config(
                config.variable_network_parameters,
                config.variable_data_parameters,
                config.data_parameters,
                config.network_parameters,
                config_file2.name,
            )
            # load this saved file
            config2 = load_config(config_file2.name)
            # check that all parameters are still the same
            assert config2.variable_data_parameters == variable_data_parameters
            assert config2.variable_network_parameters == variable_network_parameters
            assert config2.network_parameters == network_parameters
            assert config2.data_parameters == data_parameters


@pytest.mark.slow
def test_run_experiment():
    # first, train model on different folds, iterations (data_parameters) and learning rate (network_parameters)
    data_parameters = get_data_parameters()
    data_parameters["number_of_folds"] = 10
    data_parameters["fold_n"] = [0, 9]
    variable_data_parameters = {"iteration": [0, 1]}
    network_parameters = get_network_parameters()
    variable_network_parameters = {"learning_rate": [0.001, 0.005]}

    # create config file
    with tempfile.NamedTemporaryFile(suffix=".py") as config_file:
        create_config_file(
            fname=config_file.name,
            data_parameters=data_parameters,
            network_parameters=network_parameters,
            variable_data_parameters=variable_data_parameters,
            variable_network_parameters=variable_network_parameters,
        )

        # create temporary experiment path
        with tempfile.TemporaryDirectory() as experiment_path:
            print("calling")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            script_path = os.path.abspath(os.path.join(dir_path, "../../scripts/classifier/run.py"))
            print(script_path)
            subprocess.run(
                [
                    "python",
                    script_path,
                    "train",
                    "--no-optimise-threshold",
                    "--config-file",
                    config_file.name,
                    "--base-experiment-path",
                    experiment_path,
                ]
            )
            # check if the expected folder structure was created
            for lr in ["0.001", "0.005"]:
                for fold in ["0", "9"]:
                    exp = os.path.join(
                        experiment_path, "learning_rate_{}".format(network_parameters["date"]), "fold_{}".format(fold)
                    )
                    assert Experiment.exists_experiment(exp, "learning_rate_{}".format(lr))
            for i in ["0", "1"]:
                for fold in ["0", "9"]:
                    exp = os.path.join(
                        experiment_path, "iteration_{}".format(network_parameters["date"]), "fold_{}".format(fold)
                    )
                    assert Experiment.exists_experiment(exp, "iteration_{}".format(i))

            # now, try to evaluate
            subprocess.run(
                [
                    "python",
                    script_path,
                    "eval",
                    "--no-optimise-threshold",
                    "--config-file",
                    config_file.name,
                    "--base-experiment-path",
                    experiment_path,
                ]
            )
