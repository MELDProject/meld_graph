import torch_geometric.data
from meld_classifier.meld_cohort import MeldSubject
from meld_classifier.dataset import load_combined_hemisphere_data
from meld_classifier.data_preprocessing import Preprocess
import numpy as np
import torch

class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, subject_ids, cohort, params, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.params = params
        self.subject_ids = subject_ids
        self.cohort = cohort

        # preload data in memory, with all preprocessing done
        self.data_list = []
        prep = Preprocess(cohort=self.cohort)
        for subj_id in self.subject_ids:
            features_left, lesion_left, features_right, lesion_right = prep.get_data_preprocessed(subject=subj_id, features=self.params['features'], 
                params=self.params['preprocessing_params'])
            if self.params['combine_hemis'] == 'stack':
                features = np.hstack([features_left, features_right])
                lesion = np.hstack([lesion_left, lesion_right])
                self.data_list.append((features, lesion))

                features = np.hstack([features_right, features_left])
                lesion = np.hstack([lesion_right, lesion_left])
                self.data_list.append((features, lesion))
            else:
                raise NotImplementedError



    @classmethod
    def from_experiment(cls, experiment, mode):
        # set subject_ids
        train_ids, val_ids, test_ids = experiment.get_train_val_test_ids()
        if mode == "train":
            subject_ids = train_ids
        elif mode == "val":
            subject_ids = val_ids
        elif mode == "test":
            subject_ids = test_ids
        else:
            raise NotImplementedError(mode)
        # ensure that features are saved in experiment.data_parameters
        experiment.get_features()
        return cls(
            subject_ids=subject_ids,
            cohort=experiment.cohort,
            params=experiment.data_parameters,
        )

    def len(self):
        # every subject will be shown twice per epoch
        return 2*len(self.subject_ids)
    
    def get(self, idx):
        print('dataset get idx ', idx)
        features, labels = self.data_list[idx]
        return torch_geometric.data.Data(
            x=torch.tensor(features, dtype=torch.float), 
            y=torch.tensor(labels, dtype=torch.long), 
            num_nodes=len(features))

