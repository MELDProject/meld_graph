import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import numpy as np
import scipy
import json
import pandas as pd


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
        
        # TODO : Update subjects_ids list
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

    def evaluate(self,):
        """
        Evaluate the model.
        Runs `self.get_metrics(); self.plot_prediction_space(); self.plot_subjects_prediction()`
        and saves images to results folder.
        """
        # need to load and predict data
        if self.data_dictionary is None:
            self.load_predict_data()
        # calculate stats 
        self.stat_subjects()
        # make images if asked for
        if self.make_images:
            self.plot_subjects_prediction()


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
        #TODO: enable batch_size > 1
        data_loader = torch_geometric.loader.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
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


    def stat_subjects(self, suffix="", fold=None):
        """calculate stats for each subjects
        """
        
        #TODO: need to add boundaries and clusters
        # boundary_label = MeldSubject(subject, self.experiment.cohort).load_boundary_zone(max_distance=20)
        
        # columns: ID, group, detected,  number extra-lesional clusters,border detected
        # calculate stats first
        patient_dice_vars = {"TP": 0, "FP": 0, "FN": 0}
        for subject in self.data_dictionary.keys():
            prediction = self.data_dictionary[subject]["result"]
            labels = self.data_dictionary[subject]["input_labels"]
            group = labels.sum()!= 0
            
            detected = np.logical_and(prediction, labels).any()
            difference = np.setdiff1d(np.unique(prediction), np.unique(prediction[labels]))
            difference = difference[difference > 0]
            n_clusters = len(difference)
        # # if not detected, does a cluster overlap boundary zone and if so, how big is the cluster?
        # if not detected and prediction[np.logical_and(boundary_label, ~labels)].sum() > 0:
        #     border_verts = prediction[np.logical_and(boundary_label, ~labels)]
        #     i, counts = np.unique(border_verts, return_counts=True)
        #     counts = counts[i > 0]
        #     i = i[i > 0]
        #     cluster_index = i[np.argmax(counts)]
        #     border_detected = np.sum(prediction == cluster_index)
        # else:
        #     border_detected = 0

            if group == 1:
                mask = prediction.astype(bool)
                label = labels.astype(bool)
                patient_dice_vars["TP"] += np.sum(mask * label)
                patient_dice_vars["FP"] += np.sum(mask * ~label)
                patient_dice_vars["FN"] += np.sum(~mask * label)
            
            sub_df = pd.DataFrame(
                np.array([subject, group, detected, n_clusters, patient_dice_vars["TP"], patient_dice_vars["FP"], patient_dice_vars["FN"]]).reshape(-1, 1).T,
                columns=["ID", "group", "detected", "n_clusters", 'dice_tp', 'dice_fp', 'dice_fn' ],
            )
            filename = os.path.join(self.save_dir, "results", f"test_results{suffix}.csv")
            if fold is not None:
                filename = os.path.join(self.save_dir, "results", f"test_results_{fold}{suffix}.csv")

            if os.path.isfile(filename):
                done = False
                while not done:
                    try:
                        df = pd.read_csv(filename, index_col=False)
                        # df = df.append(sub_df, ignore_index=True)
                        df = pd.concat([df, sub_df], ignore_index=True, sort=False)
                        df.to_csv(filename, index=False)
                        done = True
                    except pd.errors.EmptyDataError:
                        done = False
            else:
                sub_df.to_csv(filename, index=False)

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
                os.makedirs(os.path.join(self.save_dir, "results", "images",), exist_ok=True)

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
