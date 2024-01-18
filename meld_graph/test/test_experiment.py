# #### tests for experiment.py ####
# # this is a long test, if want to exclude it for quick testing, run
# # pytest -m "not slow"
# # tested functions:
# #   _train_val_test_split_folds - test reproducibility of splitting in folds
# #   train_network - is running / creates expected outputs / can evaluate -> part of these tests are in test_script_run

## TODO: need to adapt as not working with meld_graph

# from meld_graph.experiment import Experiment
# from meld_graph.meld_cohort import MeldCohort
# import os
# import pytest
# import datetime
# from meld_graph.paths import DEFAULT_HDF5_FILE_ROOT


# @pytest.fixture(scope="session")
# def experiment(tmpdir_factory):
#     data_parameters = {
#         "harmo code": ["TEST"],
#         "scanners": ["15T", "3T"],
#         "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
#         "dataset": "MELD_dataset_TEST.csv",
#         "group": "both",
#         "features_to_exclude": [],
#         "subject_features_to_exclude": [""],
#         "number_of_folds": 5,
#         "fold_n": [0],
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

#     network_parameters = {
#         ##### network architecture #####
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
#         "name": "23-10-30_graph_combat",
#         "network_type": "MoNetUnet",
#         ##### training hyper-params #####
#         "max_patience": 10,
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
#         },
#         "date": datetime.datetime.now().strftime("%y-%m-%d"),
#     }

#     experiment_path = str(tmpdir_factory.mktemp("experiment"))
#     # experiment_path = f'test_results/iteration_{network_parameters["date"]}'
#     print(experiment_path)
#     experiment = Experiment.create_with_parameters(data_parameters, network_parameters, experiment_path, "iteration_0")
#     print(experiment)
#     experiment.init_logging()
#     return experiment


# # def test_train_val_test_split_folds(experiment):
# #     c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT)
# #     subject_ids = c.get_subject_ids(site_codes=['TEST'])

# #     # get all possible splits
# #     train_ids = [[] for _ in range(5)]
# #     val_ids = [[] for _ in range(5)]
# #     test_ids = [[] for _ in range(5)]
# #     for i in range(5):
# #         train_ids[i], val_ids[i], test_ids[i] = experiment._train_val_test_split_folds(
# #             subject_ids, iteration=i, number_of_folds=5
# #         )

# #     # test ids should be the same for each fold
# #     for i in range(5):
# #         assert (test_ids[i] == test_ids[0]).all()
# #     # val ids should always be different
# #     for i in range(5):
# #         for j in range(5):
# #             if i == j:
# #                 continue
# #             assert val_ids[i].shape != val_ids[j].shape or (val_ids[i] != val_ids[j]).all()

# #     # get iteration 0 fold again
# #     train_ids0, val_ids0, test_ids0 = experiment._train_val_test_split_folds(
# #         subject_ids, iteration=0, number_of_folds=5
# #     )
# #     # train ids should be identical to train ids before
# #     assert (train_ids0 == train_ids[0]).all()


# # @pytest.mark.slow
# # def test_run_experiment(experiment):
# #     # train experiment
# #     experiment.train()

# #     # check that checkpoint files exists
# #     checkpoint_file = os.path.join(experiment.path, "models", f"{experiment.name}.index")
# #     assert os.path.isfile(checkpoint_file)
# #     # check that log file exists
# #     log_file = os.path.join(experiment.path, "logs", f"{experiment.name}.csv")
# #     assert os.path.isfile(log_file)

# #     # optimise threshold
# #     experiment.optimise_threshold()

# #     # check that created sens_sepc_curve
# #     sens_spec_curve_file = os.path.join(
# #         experiment.path, "results", "images", f"sensitivity_specificity_curve_{experiment.name}_2.png"
# #     )
# #     assert os.path.isfile(sens_spec_curve_file)

# #     # evaluate
# #     experiment.evaluate(make_images_flag=True, make_prediction_space_flag=True)

# #     # check that have created prediction space plot
# #     prediction_space_file = os.path.join(
# #         experiment.path, "results", "images", f"prediction_space_{experiment.name}.png"
# #     )
# #     assert os.path.isfile(prediction_space_file)
# #     # check that have saved test results
# #     assert os.path.isfile(os.path.join(experiment.path, "results", f"test_results_{experiment.name}.csv"))
# #     assert os.path.isfile(os.path.join(experiment.path, "results", f"per_subject_{experiment.name}_optimal.json"))
# #     assert os.path.isfile(os.path.join(experiment.path, "results", f"per_subject_{experiment.name}_0.5.json"))
# #     # check that have saved patient predictions
# #     _, val_ids, _ = experiment.get_train_val_test_ids()
# #     for val_id in val_ids:
# #         print(val_id)
# #         assert os.path.isfile(
# #             os.path.join(experiment.path, "results", "images", f"{experiment.name}_{val_id}.jpg")
# #         )
