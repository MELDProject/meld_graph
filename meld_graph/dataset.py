import torch_geometric.data
from meld_classifier.meld_cohort import MeldSubject
from meld_classifier.dataset import load_combined_hemisphere_data
from meld_graph.data_preprocessing import Preprocess
from meld_graph.icospheres import IcoSpheres
from meld_graph.models import HexPool
from meld_graph.augment import Augment
import numpy as np
import torch
import logging
import time


class Oversampler(torch.utils.data.Sampler):
    """
    NNUnet-like oversampling.
    33% lesional, 66% random.

    Shuffles data after sampling
    """

    def __init__(self, data_source):
        self.log = logging.getLogger(__name__)
        self.data_source = data_source
        self._num_samples = None

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source.lesional_idxs) * 3
        return self._num_samples

    def get_sampling_idxs(self):
        """
        return list of len num_samples, containing data_source idxs to sample.
        Call once per epoch (when restarting iterator)
        """
        n_non = len(self.data_source.lesional_idxs) * 2
        non_ids = torch.randperm(len(self.data_source), dtype=torch.int64)[:n_non]
        ids_to_choose = torch.hstack(
            [torch.from_numpy(self.data_source.lesional_idxs), non_ids]
        )
        return ids_to_choose[torch.randperm(len(ids_to_choose), dtype=torch.int64)]

    def __iter__(self):
        ids_to_choose = self.get_sampling_idxs()
        # self.log.info(f'iterating over ids: {ids_to_choose}')
        return iter(ids_to_choose.tolist())

    def __len__(self):
        return self.num_samples


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(
        self,
        subject_ids,
        cohort,
        params,
        mode="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        output_levels=[],
    ):
        """
        output_levels: list of icosphere levels for which y should be returned as well. Used for deep supervision.
            will be available as self.get().output_level<level>.
        """
        super().__init__(None, transform, pre_transform, pre_filter)
        self.log = logging.getLogger(__name__)
        self.params = params
        self.subject_ids = subject_ids
        self.cohort = cohort
        self.mode = mode
        self.augment = None
        if (self.mode != "test") & (self.params["augment_data"] != None):
            self.augment = Augment(self.params["augment_data"])
        self.output_levels = sorted(output_levels)
        if len(self.output_levels) != 0:
            self.icospheres = IcoSpheres()
            self.pool_layers = {
                level: HexPool(self.icospheres.get_neighbours(level=level))
                for level in range(min(self.output_levels), 7)[::-1]
            }
        self._lesional_idxs = None

        # preload data in memory, with all preprocessing done
        self.data_list = []
        self.prep = Preprocess(
            cohort=self.cohort, params=self.params["preprocessing_parameters"]
        )
        self.log.info(f"Loading and preprocessing {mode} data")
        self.log.debug(f"Combine hemis {self.params['combine_hemis']}")

        # switch for synthetic task here
        if self.params["synthetic_data"]["run_synthetic"]:
            self.icospheres = IcoSpheres() # TODO why instanciate again?
            # if not using controls, generate data and return
            if not self.params['synthetic_data']['use_controls']:
                self.subject_ids = np.arange(self.params['synthetic_data']['n_subs'])
                self.log.info(f"WARNING: Simulating {len(self.subject_ids)} subjects")
                for s in np.arange(self.params['synthetic_data']['n_subs']):
                    sfl, sfr, sll, slr = self.synthetic_lesion()
                    self.data_list.append((sfl.T, sll))
                    self.data_list.append((sfr.T, slr))
                return
            #undersample subject ids to get controlled number
            n_subs_before = len(self.subject_ids)
            if self.params['synthetic_data']['n_subs']<len(self.subject_ids):
                self.subject_ids = np.random.choice(self.subject_ids,self.params['synthetic_data']['n_subs'])
                self.subject_samples = np.arange(len(self.subject_ids))
            elif self.params['synthetic_data']['n_subs']>len(self.subject_ids):
                #if wanting multiple samples of same subjects
                self.subject_samples = np.sort(np.random.choice(np.arange(len(self.subject_ids)),
                                                        self.params['synthetic_data']['n_subs']))
            self.log.info(f"WARNING: Simulating {len(self.subject_samples)} subjects using {n_subs_before} controls")                                       
                
        for s_i,subj_id in enumerate(self.subject_ids):
            #load in control data
            features_left, features_right, lesion_left, lesion_right = self.prep.get_data_preprocessed(subject=subj_id, 
                                                                     features=params['features'], 
                                        lobes = params['lobes'], lesion_bias=False)
            #add lesion
            if self.params['synthetic_data']['run_synthetic']:
                for duplicate in np.arange(np.sum(self.subject_samples==s_i)):
                    sfl, sfr, sll, slr = self.synthetic_lesion(features_left.copy(),
                                                               features_right.copy(),
                                                              )
                    if self.params['combine_hemis'] is None:
                        self.data_list.append((sfl.T, sll))
                        self.data_list.append((sfr.T, slr))
            else:

                if self.params["combine_hemis"] is None:
                    self.data_list.append((features_left.T, lesion_left))
                    self.data_list.append((features_right.T, lesion_right))
                elif self.params["combine_hemis"] == "stack":
                    # this code doesn't look correct. surely lesions also should be stacked?
                    features = np.vstack([features_left, features_right]).T
                    self.data_list.append((features, lesion_left))
                    features = np.vstack([features_right, features_left]).T
                    self.data_list.append((features, lesion_right))
                else:
                    raise NotImplementedError
        #dataset has weird properties. subject_ids needs to be the right length, matching the data length
        if self.params['synthetic_data']['run_synthetic']:
            if self.params['synthetic_data']['n_subs']>len(self.subject_ids):
                self.subject_ids = np.array(self.subject_ids)[self.subject_samples]
        return

    def synthetic_lesion(self, features_left=None, features_right=None):
        """add synthetic lesion to input features for both hemis"""
        fs=[]
        ls=[]
        for f in [features_left,features_right]:
            #controls the proportion of examples with lesions.

            subtype=np.random.choice(self.params['synthetic_data']['n_subtypes'])
            f, l = self.prep.generate_synthetic_data(self.icospheres.icospheres[7]['coords'],
                                                              len(self.params['features']),
                                                              self.params['synthetic_data']['bias'],
                                                             self.params['synthetic_data']['radius'],
                                                             histo_type_seed=subtype,
                    proportion_features_abnormal=self.params['synthetic_data']['proportion_features_abnormal'],
                    proportion_hemispheres_abnormal=self.params['synthetic_data']['proportion_hemispheres_lesional'],
                                                 features=f,
                                                    jitter_factor=self.params['synthetic_data']['jitter_factor'],
                                                    smooth_lesion=self.params['synthetic_data'].get('smooth_lesion', False))
            f[:,~self.cohort.cortex_mask]=0
            l[~self.cohort.cortex_mask]=0
            fs.append(f)
            ls.append(l)
        return fs[0], fs[1], ls[0], ls[1]

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
            mode=mode,
            output_levels=experiment.network_parameters["training_parameters"]
            .get("deep_supervision", {})
            .get("levels", []),
        )

    def len(self):
        # every subject will be shown twice per epoch
        return 2 * len(self.subject_ids)

    def get(self, idx):
        # print('dataset get idx ', idx)
        features, labels = self.data_list[idx]
        # apply data augmentation
        if self.augment != None:
            features, labels = self.augment.apply(features, labels)
        data = torch_geometric.data.Data(
            x=torch.tensor(features, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.long),
            num_nodes=len(features),
        )
        if len(self.output_levels) != 0:
            # add extra output levels to data
            labels_pooled = {7: data.y}
            for level in range(min(self.output_levels), 7)[::-1]:
                labels_pooled[level] = self.pool_layers[level](labels_pooled[level + 1])
            for level in self.output_levels:
                setattr(data, f"output_level{level}", labels_pooled[level])
        return data

    @property
    def lesional_idxs(self):
        """find ids of data entries with lesional examples"""
        if self._lesional_idxs is None:
            lesional_idxs = []
            for i, d in enumerate(self.data_list):
                if d[1].sum():
                    lesional_idxs.append(i)
            self._lesional_idxs = np.array(lesional_idxs)
            # self.log.info(f'lesional idxs: {self._lesional_idxs}')
        return self._lesional_idxs
