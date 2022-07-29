from meld_graph.paths import EXPERIMENT_PATH
import json
import logging
import os
import torch
import meld_graph.models
from meld_classifier.meld_cohort import MeldCohort
from meld_graph.training import Trainer
import numpy as np
import pandas as pd
import glob

def is_experiment(path, trained=False):
    """
    convenience function to check that this is a valid experiment folder, containing network and data parameters.
    If trained is set to True, also needs to contain best_model.pt and train/val scores
    """
    if os.path.isfile(os.path.join(path, 'data_parameters.json')) and os.path.isfile(os.path.join(path, 'network_parameters.json')):
        # is experiment folder
        if trained is False:
            return True
        else:
            if os.path.isfile(os.path.join(path, 'best_model.pt')) and \
            os.path.isfile(os.path.join(path, 'train_scores.csv')) and \
            os.path.isfile(os.path.join(path, 'val_scores.csv')):
                return True
    return False

def discover_trained_experiments(path=None):
    """
    recursively search for experiment folders starting from path
    """
    if path is None:
        path = EXPERIMENT_PATH
    if is_experiment(path, trained=True):
        return [path]
    else:
        experiments = []
        for f in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(f):
                experiments.extend(discover_trained_experiments(f))
        return experiments


class Experiment:
    def __init__(self, network_parameters, data_parameters, save_params=True, verbose=logging.WARNING):
        self.network_parameters = network_parameters
        self.data_parameters = data_parameters
        self.model = None # loaded by self.load_model()
        self.experiment_name = self.network_parameters['name']
        self.fold = self.data_parameters['fold_n']
        self.experiment_path = None
        if self.experiment_name is not None:
            self.experiment_path = os.path.join(EXPERIMENT_PATH, self.experiment_name, f'fold_{self.fold:02d}')
            os.makedirs(self.experiment_path, exist_ok=True)
        # init logging now, path is created
        # if save_params, will overwrite/append to logs
        self._init_logging(verbose, save_params)
        self.log = logging.getLogger(__name__)

        self.cohort = MeldCohort(
            hdf5_file_root=self.data_parameters["hdf5_file_root"], dataset=self.data_parameters["dataset"]
        )
        if save_params:
            self.save_parameters()

    @classmethod
    def from_folder(cls, experiment_path):
        """experiment_path: experiment_name/fold_00 """
        data_parameters = json.load(open(os.path.join(experiment_path, "data_parameters.json")))
        network_parameters = json.load(open(os.path.join(experiment_path, "network_parameters.json")))
        return cls(network_parameters, data_parameters, save_params=False)

    def _init_logging(self, verbose, save_params=False):
        """
        Set up a logger for this experiment that logs to experiment_path and to stdout.
        Should only be called once per experiment (overwrites existing log files of the same name)
        """
        # remove all previous logging handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # set up logging handlers
        if self.experiment_path is not None and save_params:
            fname = os.path.join(self.experiment_path, "train.log")
            fileFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fileHandler = logging.FileHandler(fname, 'w')
            fileHandler.setFormatter(fileFormatter)
            handlers=[
                        fileHandler,
                        logging.StreamHandler()
                    ]
        else:
            handlers=[logging.StreamHandler()
                ]
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=handlers
        )
        # (mostly) silence tf logging
        #tf_logger = logging.getLogger("tensorflow")
        #tf_logger.setLevel(logging.ERROR)
        # (mostly) silence matplotlib logging
        #mpl_logger = logging.getLogger("matplotlib")
        #mpl_logger.setLevel(logging.WARNING)

    def save_parameters(self):
        """
        Save dictionaries to experiment_path using json
        """
        if self.experiment_path is not None:
            self.log.info(f"saving parameter files to {self.experiment_path}")
            # data_parameters
            fname = os.path.join(self.experiment_path, "data_parameters.json")
            json.dump(self.data_parameters, open(fname, "w"), indent=4)
            # network_parameters
            fname = os.path.join(self.experiment_path, "network_parameters.json")
            json.dump(self.network_parameters, open(fname, "w"), indent=4)
        else:
            self.log.info("experiment_path is None, could not save parameters")

    def get_features(self):
        """
        get list of features that model should be trained on.
        Either read from data_parameters, or calculated and written to data_parameters
        """
        if "features" not in self.data_parameters:
            self.log.info("get features to train on")
            # get features
            features = self.cohort.get_features(features_to_exclude=self.data_parameters["features_to_exclude"])
            # get features that should be ignored
            _, features_to_ignore = self.cohort._filter_features(
                features_to_exclude=self.data_parameters.get("features_to_replace_with_0", []), return_excluded=True
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
        return self.data_parameters["features"], self.data_parameters["features_to_replace_with_0"]

    def load_model(self, checkpoint_path=None, force=False):
        """
        build model and optionally load weights from checkpoint

        checkpoint_path: absolute path to checkpoint
        """
        if self.model is not None and not force:
            self.log.info("model already exists. Specify force=True to force reloading and initialisation")

        # get number of features - depends on how hemis are combined
        if self.data_parameters['combine_hemis'] is None:
            num_features = len(self.get_features()[0])
        elif self.data_parameters['combine_hemis'] == 'stack':
            num_features = len(self.get_features()[0])*2
        else:
            raise NotImplementedError(self.data_parameters['combine_hemis'])
        # build model using network_parameters
        network_type = self.network_parameters['network_type']
        # build icosphere_params dict
        icosphere_params = self.data_parameters['icosphere_parameters']
        icosphere_params['combine_hemis'] = self.data_parameters['combine_hemis']
        icosphere_params['conv_type'] = self.network_parameters['model_parameters']['conv_type']
        if network_type == 'MoNet':
            self.model = meld_graph.models.MoNet(**self.network_parameters['model_parameters'], num_features=num_features, icosphere_params=icosphere_params)
        elif network_type == 'MoNetUnet':
            self.model = meld_graph.models.MoNetUnet(**self.network_parameters['model_parameters'], num_features=num_features, 
                icosphere_params=icosphere_params, deep_supervision=self.network_parameters['training_parameters'].get('deep_supervision', {}).get('levels', []))
        elif network_type == 'SimpleNet':
            self.model = meld_graph.models.SimpleNet(**self.network_parameters['model_parameters'], num_features=num_features,
                icosphere_params=icosphere_params)
        else:
            raise(NotImplementedError, network_type)
        
        # TODO below code is unchecked
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            # checkpoint contains both model architecture + weights
            self.log.info(f"Loading model weights from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.eval()

    def train(self):
        trainer = Trainer(self)
        trainer.train()

    def get_train_val_test_ids(self):
        """
        return train val test ids.
        Either read from data_parameters (if exist), or created using _train_val_test_split_folds.

        returns train_ids, val_ids, test_ids
        """
        if "train_ids" not in self.data_parameters:
            self.log.info("getting train val test split")
            # get subject ids restricted to desired subjects
            subject_ids = self.cohort.get_subject_ids(**self.data_parameters)
            # get train val test split
            train_ids, val_ids, test_ids = self._train_val_test_split_folds(
                subject_ids,
                iteration=self.data_parameters["fold_n"],
                number_of_folds=self.data_parameters["number_of_folds"],
            )
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
        return self.data_parameters["train_ids"], self.data_parameters["val_ids"], self.data_parameters["test_ids"]

    def _train_val_test_split_folds(self, subject_ids, iteration=0, number_of_folds=10):
        """split subject_ids into train val and test.
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

        _, dataset_trainval_ids, dataset_test_ids = self.cohort.read_subject_ids_from_dataset()
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

    def get_scores(self, split='val'):
        print(self.experiment_path)
        if is_experiment(self.experiment_path, trained=True):
            df = pd.read_csv(os.path.join(self.experiment_path, f'{split}_scores.csv'), index_col=0)
            return df
        else:
            self.log.info('Experiment is not trained, no scores available')
            return None
