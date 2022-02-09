import torch_geometric.data
from meld_classifier.meld_cohort import MeldSubject
from meld_classifier.dataset import load_combined_hemisphere_data
import torch

class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, subject_ids, cohort, params, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.params = params
        self.subject_ids = subject_ids
        self.cohort = cohort
        # TODO could preload data here

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
        return len(self.subject_ids)
    
    def get(self, idx):
        subj_id = self.subject_ids[idx]
        subj = MeldSubject(subj_id, self.cohort)
        # TODO use preprocessing code from Mathilde and Sophie here
        features, labels = load_combined_hemisphere_data(
                subj,
                self.params['features'],
                features_to_ignore=self.params['features_to_replace_with_0'],
                universal_features=[],
                demographic_features=[],
                num_neighbours=0,
                normalise=False,
            )
        return torch_geometric.data.Data(
            x=torch.tensor(features, dtype=torch.float), 
            y=torch.tensor(labels, dtype=torch.long), 
            num_nodes=len(features))

