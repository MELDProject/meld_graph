## Script to calculate the mean and standard deviation of the MELD cohort surface-based features
## Parameters are saved and used for normalisation

from meld_graph.data_preprocessing import Preprocess as Prep
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import numpy as np
import json


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = (
                m / (m + n) * self.std**2 + n / (m + n) * newstd**2 + m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            )
            self.std = np.sqrt(self.std)

            self.nobservations += n


if __name__ == "__main__":

    config = load_config("config_files/example_experiment_config.py")
    cohort = MeldCohort(
        hdf5_file_root=config.data_parameters["hdf5_file_root"],
        dataset=config.data_parameters["dataset"],
    )
    prep = Prep(cohort=cohort, params=config.data_parameters)
    subject_ids = cohort.get_subject_ids(group="both")
    # two batch-wise stats recorders, one with flair, one without
    flair_mask = np.zeros(len(config.data_parameters["features"]), dtype=bool)
    for fi, feature in enumerate(config.data_parameters["features"]):
        if "FLAIR" in feature:
            flair_mask[fi] = 1

    mean_std = StatsRecorder()
    mean_std_flair = StatsRecorder()

    for si, subj_id in enumerate(subject_ids):
        if si % 30 == 0:
            print(f"{100*si/len(subject_ids)}% complete")

        subject_data_list = prep.get_data_preprocessed(
            subject=subj_id,
            features=config.data_parameters["features"],
            lobes=config.data_parameters["lobes"],
            lesion_bias=False,
        )
        for hemisphere_data in subject_data_list:
            feat_hem = hemisphere_data["features"].T
            feat_hem = feat_hem[:, cohort.cortex_mask]
            feat_hem_nf = feat_hem[~flair_mask]
            mean_std.update(feat_hem_nf.T)
            if np.sum(feat_hem[6]) != 0:
                feat_hem_f = feat_hem[flair_mask]
                mean_std_flair.update(feat_hem_f.T)

    means_stds = np.zeros((2, len(config.data_parameters["features"])))
    means_stds[0, flair_mask] = mean_std_flair.mean
    means_stds[1, flair_mask] = mean_std_flair.std
    means_stds[0, ~flair_mask] = mean_std.mean
    means_stds[1, ~flair_mask] = mean_std.std

    mean_stds_dict = {}
    for fi, feature in enumerate(config.data_parameters["features"]):
        mean_stds_dict[feature] = {}
        mean_stds_dict[feature]["mean"] = means_stds[0, fi]
        mean_stds_dict[feature]["std"] = means_stds[1, fi]

    means_stds = np.zeros((2,len(config.data_parameters['features'])))
    means_stds[0,flair_mask] = mean_std_flair.mean
    means_stds[1,flair_mask] = mean_std_flair.std
    means_stds[0,~flair_mask] = mean_std.mean
    means_stds[1,~flair_mask] = mean_std.std

    mean_stds_dict={
                   }
    for fi,feature in enumerate(config.data_parameters['features']):
        mean_stds_dict[feature]={}
        mean_stds_dict[feature]['mean'] = means_stds[0,fi]
        mean_stds_dict[feature]['std'] = means_stds[1,fi]

    with open('../data/feature_means_no_combat.json', 'w') as fp:
        json.dump(mean_stds_dict, fp)
