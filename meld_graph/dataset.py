import torch_geometric.data
from meld_classifier.meld_cohort import MeldSubject
from meld_classifier.dataset import load_combined_hemisphere_data
from meld_graph.data_preprocessing import Preprocess
from meld_graph.augment import Augment
import numpy as np
import torch
import logging
import time

class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, subject_ids, cohort, params, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.log = logging.getLogger(__name__)
        self.params = params
        self.subject_ids = subject_ids
        self.cohort = cohort
        self._augment =  None

        # preload data in memory, with all preprocessing done
        self.data_list = []
        prep = Preprocess(cohort=self.cohort)
        self.log.info("Loading and preprocessing data")
        self.log.info(f"Combine hemis {self.params['combine_hemis']}")
        for subj_id in self.subject_ids:
            features_left, features_right, lesion_left, lesion_right = prep.get_data_preprocessed(subject=subj_id, features=self.params['features'], 
                params=self.params['preprocessing_parameters'], lobes = self.params['lobes'])
            if self.params['combine_hemis'] is None:
                self.data_list.append((features_left.T, lesion_left))
                self.data_list.append((features_right.T, lesion_right))
            elif self.params['combine_hemis'] == 'stack':
                features = np.vstack([features_left, features_right]).T
                self.data_list.append((features, lesion_left))

                features = np.vstack([features_right, features_left]).T
                self.data_list.append((features, lesion_right))
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

    
    @property
    def augment(self):
        if self._augment is None:
            self._augment = Augment(self.params['augment_data']) 
        return self._augment
    
    def len(self):
        # every subject will be shown twice per epoch
        return 2*len(self.subject_ids)
    
    def get(self, idx):
        #print('dataset get idx ', idx)
        features, labels = self.data_list[idx]
        #apply data augmentation
        if self.params['augment_data'] != None:
            features, labels = self.augment.apply(features, labels)
        
        return torch_geometric.data.Data(
            x=torch.tensor(features, dtype=torch.float), 
            y=torch.tensor(labels, dtype=torch.long), 
            num_nodes=len(features))

