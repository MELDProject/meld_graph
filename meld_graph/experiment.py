from meld_graph.paths import EXPERIMENT_PATH
import json
import logging
import os
import torch
import meld_graph.models
from meld_graph.meld_cohort import MeldCohort
from meld_graph.ensemble import Ensemble
from meld_graph.training import Trainer
import numpy as np
import pandas as pd
import copy
import glob


def is_experiment(path, trained=False):
    """
    Convenience function to check that this is a valid experiment folder, containing network and data parameters.

    Args:
        trained (bool): experiments needs to contain best_model.pt and train/val scores

    Returns:
        True if path contained valid experiment
    """
    if os.path.isfile(os.path.join(path, "data_parameters.json")) and os.path.isfile(
        os.path.join(path, "network_parameters.json")
    ):
        # is experiment folder
        if trained is False:
            return True
        else:
            if (
                os.path.isfile(os.path.join(path, "best_model.pt"))
                and os.path.isfile(os.path.join(path, "train_scores.csv"))
                and os.path.isfile(os.path.join(path, "val_scores.csv"))
            ):
                return True
    return False


def discover_trained_experiments(path=None):
    """
    Recursively search for experiment folders starting from path.

    Args:
        path (optional, str): path containing experiment folders. Default: EXPERIMENT_PATH.
    """
    if path is None:
        path = EXPERIMENT_PATH
    if is_experiment(path, trained=True):
        return [path]
    else:
        experiments = []
        for f in glob.glob(os.path.join(path, "*")):
            if os.path.isdir(f):
                experiments.extend(discover_trained_experiments(f))
        return experiments


class Experiment:
    """
    Experiment class for setting up experiments and loading models.

    Args:
        network_parameters (dict): parameters for setting up model and training. See example_experiment_config.py for options.
        data_parameters (dict): parameters for setting up dataset. See example_experiment_config.py for options.
        save_params (bool): save network and data parameters as json files in experiment folder.
        verbose (int): logging level.
    """

    def __init__(
        self,
        network_parameters,
        data_parameters,
        save_params=True,
        verbose=logging.INFO,
    ):
        self.network_parameters = network_parameters
        self.data_parameters = data_parameters
        self.model = None  # loaded by self.load_model()
        self.experiment_name = self.network_parameters["name"]
        self.fold = self.data_parameters["fold_n"]
        self.experiment_path = None
        if self.experiment_name is not None:
            if isinstance(self.fold, int) :
                self.experiment_path = os.path.join(EXPERIMENT_PATH, self.experiment_name, f"fold_{self.fold:02d}")
            else:
                self.experiment_path = os.path.join(EXPERIMENT_PATH, self.experiment_name, f"fold_{self.fold}")
            os.makedirs(self.experiment_path, exist_ok=True)
        # init logging now, path is created
        # if save_params, will overwrite/append to logs
        self._init_logging(verbose, save_params)
        self.log = logging.getLogger(__name__)

        self.cohort = MeldCohort(
            hdf5_file_root=self.data_parameters["hdf5_file_root"],
            dataset=self.data_parameters["dataset"],
        )
        if save_params:
            self.save_parameters()
        self.log.info(f"Initialised Experiment {self.experiment_name}")

    @classmethod
    def from_folder(cls, experiment_path):
        """
        Set up experiment for existing experiment_path.

        Args:
            experiment_path (str): path to experiment. E.g. experiment_name/fold_00
        """
        data_parameters = json.load(open(os.path.join(experiment_path, "data_parameters.json")))
        network_parameters = json.load(open(os.path.join(experiment_path, "network_parameters.json")))
        return cls(network_parameters, data_parameters, save_params=False)

    def _init_logging(self, verbose, save_params=False):
        """
        Set up a logger for this experiment that logs to experiment_path and to stdout.
        Should only be called once per experiment (overwrites existing log files of the same name).

        Args:
            verbose (int): logging level.
            save_params (bool): if true, also log to experiment_path.
        """
        # remove all previous logging handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # set up logging handlers
        if self.experiment_path is not None and save_params:
            fname = os.path.join(self.experiment_path, "train.log")
            fileFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fileHandler = logging.FileHandler(fname, "w")
            fileHandler.setFormatter(fileFormatter)
            handlers = [fileHandler, logging.StreamHandler()]
        else:
            handlers = [logging.StreamHandler()]
        logging.basicConfig(
            level=verbose,
            format="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        # (mostly) silence tf logging
        # tf_logger = logging.getLogger("tensorflow")
        # tf_logger.setLevel(logging.ERROR)
        # (mostly) silence matplotlib logging
        # mpl_logger = logging.getLogger("matplotlib")
        # mpl_logger.setLevel(logging.WARNING)

    def save_parameters(self):
        """
        Save dictionaries to experiment_path using json.
        """
        if self.experiment_path is not None:
            self.log.info(f"Saving parameter files to {self.experiment_path}")
            # data_parameters
            fname = os.path.join(self.experiment_path, "data_parameters.json")
            json.dump(self.data_parameters, open(fname, "w"), indent=4)
            # network_parameters
            fname = os.path.join(self.experiment_path, "network_parameters.json")
            json.dump(self.network_parameters, open(fname, "w"), indent=4)
        else:
            self.log.info("Experiment_path is None, could not save parameters")

    def get_features(self):
        """
        Get list of features that model should be trained on.
        Either read from data_parameters, or calculated and written to data_parameters.

        Returns:
            features, features_to_replace_with_0
        """
        if "features" not in self.data_parameters:
            self.log.info("get features to train on")
            # get features
            features = self.cohort.get_features(features_to_exclude=self.data_parameters["features_to_exclude"])
            # get features that should be ignored
            _, features_to_ignore = self.cohort._filter_features(
                features_to_exclude=self.data_parameters.get("features_to_replace_with_0", []),
                return_excluded=True,
            )
            self.log.debug(f"features {features}")
            self.log.debug(f"features_to_ignore {features_to_ignore}")
            
            # put train_ids, val_ids, test_ids, features in data_parameters
            self.data_parameters.update(
                {
                    "features": features,
                    "features_to_replace_with_0": features_to_ignore,
                }
            )
            # save updated data_parameters
            self.save_parameters()
        return (
            self.data_parameters["features"],
            self.data_parameters["features_to_replace_with_0"],
        )

    def load_model(self, checkpoint_path=None, force=False):
        """
        Build model and optionally load weights from checkpoint.

        Args:
            checkpoint_path (str): absolute path to model checkpoint.
            force (bool): reload model if model is already loaded.
        """
        # check if need to use load_ensemble_model
        if checkpoint_path is not None and 'fold_all' in checkpoint_path:
            return self.load_ensemble_model(checkpoint_path=checkpoint_path, force=force)
        
        if self.model is not None and not force:
            self.log.info("Model already exists. Specify force=True to force reloading and initialisation")
        self.log.info("Creating model")
        # get number of features - depends on how hemis are combined
        if self.data_parameters["combine_hemis"] is None:
            num_features = len(self.get_features()[0])
        elif self.data_parameters["combine_hemis"] == "stack":
            num_features = len(self.get_features()[0]) * 2
        else:
            raise NotImplementedError(self.data_parameters["combine_hemis"])
        # build model using network_parameters
        network_type = self.network_parameters["network_type"]
        # build icosphere_params dict
        icosphere_params = self.data_parameters["icosphere_parameters"]
        icosphere_params["combine_hemis"] = self.data_parameters["combine_hemis"]
        icosphere_params["conv_type"] = self.network_parameters["model_parameters"]["conv_type"]
        if network_type == "MoNet":
            self.model = meld_graph.models.MoNet(
                **self.network_parameters["model_parameters"],
                num_features=num_features,
                icosphere_params=icosphere_params,
            )
        elif network_type == "MoNetUnet":
            self.model = meld_graph.models.MoNetUnet(
                **self.network_parameters["model_parameters"],
                num_features=num_features,
                icosphere_params=icosphere_params,
                deep_supervision=self.network_parameters["training_parameters"]
                .get("deep_supervision", {})
                .get("levels", []),
                classification_head=self.network_parameters["training_parameters"]["loss_dictionary"]
                .get("lesion_classification", {})
                .get("apply_to_bottleneck", False),
                object_detection_head=self.network_parameters['training_parameters']['loss_dictionary']
                .get('object_detection', {})
                .get('apply_to_bottleneck', False),


            )
        else:
            raise (NotImplementedError, network_type)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            # checkpoint contains both model architecture + weights
            self.log.info(f"Loading model weights from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            self.model.eval()
        elif checkpoint_path is not None:
            self.log.warn(f"Model checkpoing {checkpoint_path} does not exist!!!")
        self.model.to(device)

    def load_ensemble_model(self, checkpoint_path=None, force=False):
        if self.model is not None and not force:
            self.log.info("Model already exists. Specify force=True to force reloading and initialisation")
        # create model without checkpoint
        self.load_model(checkpoint_path=None, force=force)
        self.log.info('Creating ensemble model')
        models = [copy.deepcopy(self.model) for _ in range(5)]  # TODO this assumes that we are always ensembling 5 models
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        for model in models:
            model.to(device)
        ensemble_model = Ensemble(models)
        self.model = ensemble_model
        # load weights from checkpoint    
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            # checkpoint contains both model architecture + weights
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.log.info(f"Loading ensemble model weights from checkpoint {checkpoint_path}")
            res = self.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            self.log.debug(f'Loading returns: {res}')
            self.model.eval()


    def train(self, wandb_logging=False):
        """
        Train model.

        Args:
            wandb_logging (bool): Log to wandb, requires active login and setup of wandb.
        """
        trainer = Trainer(self)
        trainer.train(wandb_logging=wandb_logging)

    def get_train_val_test_ids(self):
        """
        Return train val test ids.
        Either read from data_parameters (if exist), or created using _train_val_test_split_folds.

        Returns:
            train_ids, val_ids, test_ids
        """
        if "train_ids" not in self.data_parameters:
            self.log.info("Getting train val test split")
            # get subject ids restricted to desired subjects
            subject_ids = self.cohort.get_subject_ids(**self.data_parameters)
            # get train val test split
            train_ids, val_ids, test_ids = self._train_val_test_split_folds(
                subject_ids,
                iteration=self.data_parameters["fold_n"],
                number_of_folds=self.data_parameters["number_of_folds"],
            )
            self.data_parameters.get('subsample_cohort_fraction',False)
            print('orig:',len(train_ids))
            if self.data_parameters["subsample_cohort_fraction"]:
                n_total = len(train_ids)
                n_subs = np.round(self.data_parameters["subsample_cohort_fraction"] *n_total).astype(int)
                #shuffle array to keep consistent over fractions
                rng = np.random.default_rng(0)
                all_ids = np.arange(n_total)
                rng.shuffle(all_ids)
                sub_indices = all_ids[:n_subs]
                train_ids = train_ids[sub_indices]
            print('filtered:',len(train_ids))
            # put in data_parameters
            self.data_parameters.update(
                {
                    "train_ids": list(train_ids),
                    "test_ids": list(test_ids),
                    "val_ids": list(val_ids),
                }
            )
            # save updated data_parameters
            self.save_parameters()
        return (
            self.data_parameters["train_ids"],
            self.data_parameters["val_ids"],
            self.data_parameters["test_ids"],
        )

    def _train_val_test_split_folds(self, subject_ids, iteration=0, number_of_folds=10):
        """
        Split subject_ids into train val and test.

        test_ids are defined in dataset_name.
        The remaining ids are split randomly (but with a fixed seed) in number_of_folds folds.

        Args:
            list_ids (list of str): subject ids to split
            number_of_folds (int): number of folds to split the train/val ids into
            iteration (int): number of validation fold, values 0,..,number_of_folds-1
        Returns:
            train_ids, val_ids, test_ids
        """
        np.random.seed(0)

        (
            _,
            dataset_trainval_ids,
            dataset_test_ids,
        ) = self.cohort.read_subject_ids_from_dataset()
        subject_ids = np.array(subject_ids)

        # get test_ids
        test_mask = np.in1d(subject_ids, dataset_test_ids)
        test_ids = subject_ids[test_mask]

        # get trainval_ids
        trainval_ids = subject_ids[~test_mask]
        trainval_ids = np.intersect1d(trainval_ids, dataset_trainval_ids)
        # split trainval_ids in folds
        np.random.shuffle(trainval_ids)
        folds = np.array_split(trainval_ids, number_of_folds)
        folds = np.roll(folds, shift=iteration, axis=0)
        train_ids = np.concatenate(folds[0:-1]).ravel()
        val_ids = folds[-1]
        return train_ids, val_ids, test_ids

    def get_scores(self, split="val"):
        """
        Read scores from split_scores.csv and return dataframe.
        """
        print(self.experiment_path)
        if is_experiment(self.experiment_path, trained=True):
            df = pd.read_csv(os.path.join(self.experiment_path, f"{split}_scores.csv"), index_col=0)
            return df
        else:
            self.log.info("Experiment is not trained, no scores available")
            return None