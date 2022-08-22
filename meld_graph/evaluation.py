import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import numpy as np
import scipy
import json


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class Evaluator:
    """ """

    def __init__(
        self,
        experiment,
        mode="test",
        checkpoint_path=None,
        make_images=False,
        subject_ids=None,
        test_parameters=None,
        save_dir=None,
    ):

        # set class params
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        assert mode in (
            "test",
            "val",
            "train",
            "inference",
        ), "mode needs to be either test or val or train or inference"
        self.mode = mode
        self.make_images = make_images
        self.test_parameters = test_parameters
        self.subject_ids = subject_ids

        self.data_dictionary = None

        # TODO: add clustering and thershold
        # self.threshold = self.experiment.network_parameters["optimal_threshold"]
        # if threshold was not optimized, use 0.5
        # if not isinstance(self.threshold, float):
        #     self.threshold = 0.5
        # self.min_area_threshold = self.experiment.data_parameters["min_area_threshold"]
        # self.log.info("Evalution {}, {}".format(self.mode, self.threshold))

        # Initialised directory to save results and plots
        if save_dir is None:
            self.save_dir = self.experiment.path
        else:
            self.save_dir = save_dir

        # load test parameters and save in file
        if test_parameters != None:
            self.experiment.data_parameters = load_config(
                self.test_parameters
            ).data_parameters
            self.experiment.network_parameters = load_config(
                self.test_parameters
            ).network_parameters
            # Save experiment parameters:
            save_json(
                os.path.join(self.save_dir, "network_parameters.json"),
                self.experiment.network_parameters,
            )
            save_json(
                os.path.join(self.save_dir, "data_parameters.json"),
                self.experiment.data_parameters,
            )
            # TODO : Update subjects_ids list

        else:
            self.data_parameters = self.experiment.data_parameters
            self.network_parameters = self.experiment.network_parameters

        # get subject_ids
        train_ids, val_ids, test_ids = self.experiment.get_train_val_test_ids()
        if mode == "train":
            subject_ids = train_ids
        elif mode == "val":
            subject_ids = val_ids
        elif mode == "test":
            subject_ids = test_ids
        self.patient_ids, self.control_ids = self.divide_subjects(
            subject_ids, n_controls=5
        )
        self.combined_ids = list(self.patient_ids) + list(self.control_ids)

        # if checkpoint load model
        if checkpoint_path:
            self.experiment.load_model(
                checkpoint_path=os.path.join(checkpoint_path, "best_model.pt"),
                force=True,
            )

    def evaluate(self):
        """
        Evaluate the model.
        Runs `self.get_metrics(); self.plot_prediction_space(); self.plot_subjects_prediction()`
        and saves images to results folder.
        """
        # need to load and predict data
        if self.data_dictionary is None:
            self.load_predict_data()
        # make images if asked for
        if self.make_images:
            self.plot_subjects_prediction()

    def cluster_and_area_threshold(self, mask, island_count=0):
        """cluster predictions and threshold based on min_area_threshold
        Args:
            mask: boolean mask of the per-vertex lesion predictions to cluster"""
        n_comp, labels = scipy.sparse.csgraph.connected_components(
            self.experiment.cohort.adj_mat[mask][:, mask]
        )
        islands = np.zeros(len(mask))
        # only include islands larger than minimum size.
        for island_index in np.arange(n_comp):
            include_vec = labels == island_index
            size = np.sum(include_vec)
            if size >= self.min_area_threshold:
                island_count += 1
                island_mask = mask.copy()
                island_mask[mask] = include_vec
                islands[island_mask] = island_count
        return islands

    def threshold_and_cluster(self, data_dictionary=None):
        return_dict = data_dictionary is not None
        if data_dictionary is None:
            data_dictionary = self.data_dictionary
        for subj_id, data in data_dictionary.items():
            data["cluster_thresholded"] = {}
            predictions = self.experiment.cohort.split_hemispheres(data["result"])
            island_count = 0
            for h, hemi in enumerate(["left", "right"]):
                mask = predictions[hemi] >= self.threshold
                islands = self.cluster_and_area_threshold(
                    mask, island_count=island_count
                )
                data["cluster_thresholded"][hemi] = islands
                island_count += np.max(islands)
        if return_dict:
            return data_dictionary
        else:
            self.data_dictionary = data_dictionary

    def load_predict_data(
        self,
        subject_ids=None,
    ):
        """ """
        if subject_ids == None:
            subject_ids = self.combined_ids
        self.log.info("loading data and predicting model")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load dataset
        cohort = MeldCohort(
            hdf5_file_root=self.experiment.data_parameters["hdf5_file_root"]
        )
        dataset = GraphDataset(subject_ids, cohort, self.experiment.data_parameters)
        # predict on data
        data_loader = torch_geometric.loader.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.experiment.network_parameters["training_parameters"][
                "batch_size"
            ],
        )
        self.data_dictionary = {}
        prediction_array = []
        labels_array = []
        features_array = []
        for i, data in enumerate(data_loader):
            data = data.to(device)
            estimates = self.experiment.model(data.x)
            labels = data.y.squeeze()
            prediction = torch.argmax(estimates[0], axis=1)
            prediction_array.append(prediction.numpy())
            labels_array.append(labels.numpy())
            features_array.append(data.x.numpy())

        prediction_array = np.array(prediction_array)
        labels_array = np.array(labels_array)
        features_array = np.array(features_array)

        # concatenate left and right predictions and labels
        if self.experiment.data_parameters["combine_hemis"] is None:
            prediction_array = (
                prediction_array[:, cohort.cortex_mask]
                .flatten()
                .reshape((len(subject_ids), cohort.cortex_mask.sum() * 2))
            )
            labels_array = (
                labels_array[:, cohort.cortex_mask]
                .flatten()
                .reshape((len(subject_ids), cohort.cortex_mask.sum() * 2))
            )
            features_array = (
                features_array[:, cohort.cortex_mask, :]
                .flatten()
                .reshape(
                    (
                        len(subject_ids),
                        cohort.cortex_mask.sum() * 2,
                        features_array.shape[2],
                    )
                )
            )

        for i, subj_id in enumerate(subject_ids):
            self.data_dictionary[subj_id] = {
                "input_labels": labels_array[i],
                "result": prediction_array[i],
            }
            if self.mode != "train":
                self.data_dictionary[subj_id]["input_features"] = features_array[i]

    def get_metrics(
        self,
    ):
        metrics = Metrics(
            self.network_parameters["metrics"]
        )  # for keeping track of running metrics

        return metrics.get_aggregated_metrics()

    def plot_subjects_prediction(self, rootfile=None, flat_map=True):
        """plot predicted subjects"""
        import matplotlib.pyplot as plt
        import matplotlib_surface_plotting.matplotlib_surface_plotting as msp

        plt.close("all")

        for subject in self.data_dictionary.keys():
            if rootfile is not None:
                filename = os.path.join(rootfile.format(subject))
            else:
                filename = os.path.join(
                    self.save_dir, "results", "images", "{}.jpg".format(subject)
                )

            result = self.data_dictionary[subject]["result"]
            # thresholded = self.data_dictionary[subject]["cluster_thresholded"]
            label = self.data_dictionary[subject]["input_labels"]
            result = np.reshape(result, len(result))

            result_hemis = self.experiment.cohort.split_hemispheres(result)
            label_hemis = self.experiment.cohort.split_hemispheres(label)

            # initialise the icosphere or flat map
            if flat_map != True:
                from meld_graph.icospheres import IcoSpheres

                icos = IcoSpheres()
                ico_ini = icos.icospheres[7]
                coords = ico_ini["coords"]
                faces = ico_ini["faces"]
            else:
                import nibabel as nb
                from meld_classifier.paths import BASE_PATH

                flat = nb.load(
                    os.path.join(
                        BASE_PATH, "fsaverage_sym", "surf", "lh.full.patch.flat.gii"
                    )
                )
                coords, faces = flat.darrays[0].data, flat.darrays[1].data

            msp.plot_surf(
                coords,
                faces,
                [
                    result_hemis["left"],
                    # thresholded["left"],
                    label_hemis["left"],
                    result_hemis["right"],
                    # thresholded["right"],
                    label_hemis["right"],
                ],
                flat_map=flat_map,
                rotate=[90, 270],
                filename=filename,
                vmin=0.4,
                vmax=0.6,
            )
            plt.close("all")

    def divide_subjects(self, subject_ids, n_controls=5):
        """divide subject_ids into patients and controls
        if only trained on patients, controls are added.
        If self.mode is test, controls from test set (defined by dataset csv file) are added.
        If self.mode is train/val, the first/last n_controls are added.
        """
        if self.experiment.data_parameters["group"] == "patient":
            # get n_control ids (not in subject_ids, because training was only on patients)
            # get all valid control ids (with correct features etc)
            data_parameters_copy = self.experiment.data_parameters.copy()
            data_parameters_copy["group"] = "control"
            control_ids = self.experiment.cohort.get_subject_ids(
                **data_parameters_copy, verbose=False
            )
            # shuffle control ids
            np.random.seed(5)
            np.random.shuffle(control_ids)
            # filter controls by self.mode (make sure when mode is test, only test controls are used)
            if self.mode == "test":
                (
                    _,
                    _,
                    dataset_test_ids,
                ) = self.experiment.cohort.read_subject_ids_from_dataset()
                control_ids = np.array(control_ids)[
                    np.in1d(control_ids, dataset_test_ids)
                ]
                # select n_controls
                control_ids = control_ids[:n_controls]
            elif self.mode in ("train", "val"):
                (
                    _,
                    dataset_trainval_ids,
                    _,
                ) = self.experiment.cohort.read_subject_ids_from_dataset()
                control_ids = np.array(control_ids)[
                    np.in1d(control_ids, dataset_trainval_ids)
                ]
                # select n_controls (first n if mode is train, last n if mode is val)
                if len(control_ids) < n_controls * 2:
                    n_controls_train = len(control_ids) // 2
                    n_controls_val = len(control_ids) - n_controls_train
                else:
                    n_controls_train = n_controls_val = n_controls
                if self.mode == "train":
                    control_ids = control_ids[:n_controls_train]
                else:  # mode is val
                    control_ids = control_ids[-n_controls_val:]
                control_ids = list(control_ids)
            if len(control_ids) < n_controls:
                self.log.warning(
                    "only {} controls available for mode {} (requested {})".format(
                        len(control_ids), self.mode, n_controls
                    )
                )
            patient_ids = subject_ids
        else:
            patient_ids = []
            control_ids = []
            for subj_id in subject_ids:
                if MeldSubject(subj_id, self.experiment.cohort).is_patient:
                    patient_ids.append(subj_id)
                else:
                    control_ids.append(subj_id)
        return patient_ids, control_ids


def save_json(json_filename, json_results):
    """
    Save dictionaries to json
    """
    # data_parameters
    json.dump(json_results, open(json_filename, "w"), indent=4)
    return
