# #### tests for scripts/run.py ####
# # this is a long test, if want to exclude it for quick testing, run
# # pytest -m "not slow"
# # tested functionality:
# #   overall training / evaluation functionality
# #   creation of expected folders when iterating over network or data parameters
# #   evaluation

# ## TODO: need to adapt as not working with meld_graph

# import datetime
# import subprocess
# import tempfile
# import os
# import glob
# from meld_graph.experiment import Experiment
# import pytest
# from meld_graph.paths import DEFAULT_HDF5_FILE_ROOT,EXPERIMENT_PATH
# from meld_graph.experiment import is_experiment

# def get_data_parameters():
#     data_parameters = {
#        "harmo code": ["TEST"],
#         "scanners": ["15T", "3T"],
#         "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
#         "dataset": "MELD_dataset_TEST.csv",
#         "group": "both",
#         "features_to_exclude": [],
#         "subject_features_to_exclude": [""],
#         "subsample_cohort_fraction": False,
#         "synth_on_the_fly": False,
#         "number_of_folds": 5,
#         "fold_n": 0,
#         "iteration": 0,
#         "batch_size": 8,
#         "shuffle_each_epoch": True,
#         "universal_features": "",
#         "num_neighbours": 0,
#         "icosphere_parameters": {
#             "distance_type": "exact",
#             "combine_hemis": None,
#             "conv_type": "SpiralConv",
#             "icosphere_path":"data/icospheres/"
#             },
#         "augment_data": {
#             "augment_lesion": {"p": 0.0},
#             "blur": {"p": 0.2},
#             "brightness": {"p": 0.15},
#             "contrast": {"p": 0.15},
#             "extend_lesion": {"p": 0.0},
#             "flipping": {"file": "data/flipping/flipping_ico7_3.npy","p": 0.5},
#             "gamma": {"p": 0.15},
#             "low_res": {"p": 0.25},
#             "noise": {"p": 0.15},
#             "spinning": {"file": "data/spinning/spinning_ico7_10.npy","p": 0.2},
#             "warping": {"file": "data/warping/warping_ico7_10.npy","p": 0.2},
#             },
#         "synthetic_data": {"run_synthetic": False,},
#         "combine_hemis": None,
#         "lesion_bias": 0,
#         "lobes": False,
#         "object_detection": True,
#         "smooth_labels": False,
#         "preprocessing_parameters": {
#             "scaling": None,
#             "zscore": "feature_means_combat.json"
#             },
#     }
#     return data_parameters


# def get_network_parameters():
#     network_parameters = {
#         ##### network architecture #####
#         "model_parameters":{ 
#         "activation_fn": "leaky_relu",
#         "conv_type": "SpiralConv",
#         "dim": 2,
#         "distance_head": False,
#         "kernel_size": 3,
#         "layer_sizes": [[32,32,32],[32,32,32],
#                 [64,64,64],[64,64,64],[128,128,128],
#                 [128,128,128],[256,256,256]],
#         "norm": None,
#         "spiral_len": 7,
#         },
#         ##### training hyper-params #####
#         "training_parameters":{
#         "max_patience": 10,
#         "shuffle_each_epoch": True,
#         "start_epoch": 0,
#         "num_epochs": 1,
#         "batch_size": 8,
#         "deep_supervision": {
#             "levels": [6,5,4,3,2,1],
#             "weight": [0.5,0.25,0.125,0.0625,0.03125,0.0150765]
#             },
#         "init_weights": None,
#         "loss_dictionary": {
#             "cross_entropy": {"weight": 1},
#             "dice": {"class_weights": [0.0,1.0],"weight": 1},
#             "distance_regression": {"loss": "mae","weigh_by_gt": True,"weight": 1},
#             "lesion_classification": {"apply_to_bottleneck": True,"weight": 1},
#             "object_detection": {"apply_to_bottleneck": True,"weight": 1}
#             }, 
#         "lr_decay": 0.9,
#         "max_epochs_lr_decay": 10,
#         "metric_smoothing": False,
#         "optimiser": "sgd",
#         "optimiser_parameters": {
#             "lr": 0.0001,
#             "momentum": 0.99,
#             "nesterov": True
#         },
#         "oversampling": False,
#         "shuffle_each_epoch": True,
#         "start_epoch": 0,
#         "stopping_metric": {
#             "name": "loss",
#             "sign": 1
#         },
#         },
#         "name": "tmp_model",
#         "network_type": "MoNetUnet",
#         "date": datetime.datetime.now().strftime("%y-%m-%d")
#    }
#     return network_parameters


# def create_config_file(
#     fname, data_parameters, network_parameters):
#     with open(fname, "w") as f:
#         f.write("\n")
#         f.write("data_parameters = ")
#         f.write(repr(data_parameters))
#         f.write("\n")
#         f.write("network_parameters = ")
#         f.write(repr(network_parameters))
#         f.write("\n")


# # def test_save_load_config_file():
# #     # get data parameters
# #     data_parameters = get_data_parameters()
# #     data_parameters["number_of_folds"] = 5
# #     data_parameters["fold_n"] = [0, 4]
# #     variable_data_parameters = {"iteration": [0, 1]}
# #     network_parameters = get_network_parameters()
# #     variable_network_parameters = {"lr_decay": [0.5, 0.9]}
# #     print(network_parameters)
# #     # create config file
# #     with tempfile.NamedTemporaryFile(suffix=".py") as config_file:
# #         create_config_file(
# #             fname=config_file.name,
# #             data_parameters=data_parameters,
# #             network_parameters=network_parameters,
# #             variable_data_parameters=variable_data_parameters,
# #             variable_network_parameters=variable_network_parameters,
#         # )
#         # load that config file
#         # config = load_config(config_file.name)
#         # check that all parameters are the same that we have written
#         # assert config.variable_data_parameters == variable_data_parameters
#         # assert config.variable_network_parameters == variable_network_parameters
#         # assert config.network_parameters == network_parameters
#         # assert config.data_parameters == data_parameters
#         # # save the config file
#         # with tempfile.NamedTemporaryFile(suffix=".py") as config_file2:
#         #     save_config(
#         #         config.variable_network_parameters,
#         #         config.variable_data_parameters,
#         #         config.data_parameters,
#         #         config.network_parameters,
#         #         config_file2.name,
#         #     )
#         #     # load this saved file
#         #     config2 = load_config(config_file2.name)
#         #     # check that all parameters are still the same
#         #     assert config2.variable_data_parameters == variable_data_parameters
#         #     assert config2.variable_network_parameters == variable_network_parameters
#         #     assert config2.network_parameters == network_parameters
#         #     assert config2.data_parameters == data_parameters


# @pytest.mark.slow
# def test_run_experiment():
#     # first, train model on different folds, iterations (data_parameters) and learning rate (network_parameters)
#     data_parameters = get_data_parameters()
#     data_parameters["number_of_folds"] = 5
#     data_parameters["fold_n"] = 0
#     network_parameters = get_network_parameters()

#     # create config file
#     with tempfile.NamedTemporaryFile(suffix=".py") as config_file:
#         create_config_file(
#             fname=config_file.name,
#             data_parameters=data_parameters,
#             network_parameters=network_parameters,
#            )

#         # create temporary experiment path
#         # with tempfile.TemporaryDirectory(dir=EXPERIMENT_PATH, prefix=network_parameters["name"]) as experiment_path:
#         print("calling")
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         script_path_train = os.path.abspath(os.path.join(dir_path, "../../scripts/train.py"))
#         print(script_path_train)
#         subprocess.run(
#             [
#                 "python",
#                 script_path_train,
#                 "--config_file",
#                 config_file.name,
#             ]
#         )
#         # check if the expected folder structure was created
#         fold = data_parameters['fold_n']
#         exp = os.path.join(EXPERIMENT_PATH, network_parameters["name"], "fold_0{}".format(fold))
#         assert is_experiment(exp)

#         # now, try to evaluate
#         script_path_eval = os.path.abspath(os.path.join(dir_path, "../../scripts/evaluate_single_model.py"))
#         print(script_path_eval)
#         subprocess.run(
#             [
#                 "python",
#                 script_path_eval,
#                 "--model_path",
#                 exp,
#                 "--split",
#                 'trainval',
#             ]
#             )
#         os.rmdir(os.path.join(EXPERIMENT_PATH, network_parameters["name"]))
