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
        distance_maps=True
    ):
        """
        output_levels: list of icosphere levels for which y should be returned as well. Used for deep supervision.
            will be available as self.get().output_level<level>.
        distance_maps: 
            Flag to enable loading of geodesic distance maps. 
            Values for controls will be maximum possible value (200).
            Will be available as self.get().distance_map.
            If output_levels are defined as well, will additionally make downsampled distance maps 
            available as self.get().output_level<level>_distance_map.
        """
        super().__init__(None, transform, pre_transform, pre_filter)
        self.log = logging.getLogger(__name__)
        self.params = params
        self.subject_ids = subject_ids
        self.cohort = cohort
        self.mode = mode
        self.augment = None
        if (self.mode == "train") & (self.params["augment_data"] != None):
            self.augment = Augment(self.params["augment_data"])
        self.output_levels = sorted(output_levels)
        self.icospheres = IcoSpheres()
        if len(self.output_levels) != 0:
            self.pool_layers = {
                level: HexPool(self.icospheres.get_downsample(target_level=level))
                for level in range(min(self.output_levels), 7)[::-1]
            }
        self._lesional_idxs = None

        # preload data in memory, with all preprocessing done
        self.data_list = []
        self.prep = Preprocess(
            cohort=self.cohort, params=self.params["preprocessing_parameters"],
            icospheres = self.icospheres
        )
        # if distance maps are required, load them in this list
        if distance_maps:
            print('dataset using distance_maps')
        
        self.log.info(f"Loading and preprocessing {mode} data")
        self.log.debug(f"Combine hemis {self.params['combine_hemis']}")
        
        if self.params["synthetic_data"]["run_synthetic"]:
            if distance_maps:
                self.prep.setup_distance_solver()            
            self.n_subs_split_i = self.params['synthetic_data']['n_subs']//self.params['number_of_folds']
            if mode=='train':
                self.n_subs_split = self.n_subs_split_i*(self.params['number_of_folds']-1)
            elif mode=='val':
                self.n_subs_split = self.n_subs_split_i
            else:
                self.n_subs_split = self.params['synthetic_data']['n_subs']
                
            # if not using controls, generate data and return
            if not self.params['synthetic_data']['use_controls']:
                self.subject_ids = np.arange(self.n_subs_split)
                self.log.info(f"WARNING: Simulating {len(self.subject_ids)} subjects")
                for s in np.arange(self.n_subs_split):
                    synth_sub_data_list = self.synthetic_lesion()
                    self.data_list.extend(synth_sub_data_list)
                return
            #undersample subject ids to get controlled number
            n_subs_before = len(self.subject_ids)
            if self.n_subs_split<=len(self.subject_ids):
                self.subject_ids = np.random.choice(self.subject_ids,self.n_subs_split)
                self.subject_samples = np.arange(len(self.subject_ids))
            elif self.n_subs_split>len(self.subject_ids):
                #if wanting multiple samples of same subjects
                self.subject_samples = np.sort(np.random.choice(np.arange(len(self.subject_ids)),
                                                        self.n_subs_split))
        
            self.log.info(f"WARNING: Simulating {len(self.subject_samples)} subjects using {n_subs_before} controls")                                       
    
        for s_i,subj_id in enumerate(self.subject_ids):
            #load in (control) data
            # features are appended to list in order: left, right
            subject_data_list = self.prep.get_data_preprocessed(subject=subj_id, 
                                        features=params['features'], 
                                        lobes = params['lobes'], lesion_bias=False,
                                        distance_maps=distance_maps,
                                        combine_hemis = self.params['combine_hemis'])
            
            #add lesion
            if self.params['synthetic_data']['run_synthetic']:
                for duplicate in np.arange(np.sum(self.subject_samples==s_i)):
                        synth_sub_data_list = self.synthetic_lesion(subject_data_list)
                        self.data_list.extend(synth_sub_data_list)
            else:
                self.data_list.extend(subject_data_list)

            # load geodesic distance maps for regression task / lesion augmentation
        #dataset has weird properties. subject_ids needs to be the right length, matching the data length
        if self.params['synthetic_data']['run_synthetic']:
            if self.n_subs_split>len(self.subject_ids):
                self.subject_ids = np.array(self.subject_ids)[self.subject_samples]
        return

    def synthetic_lesion(self, subject_data_list=[{'features':None},
                                                  {'features':None}]):
        """add synthetic lesion to input features for both hemis"""
        synth_dicts=[]
        for si,sdl in enumerate(subject_data_list):
            #controls the proportion of examples with lesions.

            subtype=np.random.choice(self.params['synthetic_data']['n_subtypes'])
            synth_dict = self.prep.generate_synthetic_data(self.icospheres.icospheres[7]['coords'],
                                                              len(self.params['features']),
                                                             histo_type_seed=subtype,
                                                             synth_params = self.params['synthetic_data'],
                                                    features=sdl['features'],
                                                    distance_maps = 'distances' in sdl.keys())
            synth_dicts.append(synth_dict)
        return synth_dicts

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
            distance_maps='distance_regression' in experiment.network_parameters['training_parameters']["loss_dictionary"].keys(),
            # TODO also load distance maps when doing lesion augmentation
        )

    def len(self):
        # every subject will be shown twice per epoch
        return len(self.data_list)

    def get(self, idx):
        # print('dataset get idx ', idx)
        subject_data_dict = self.data_list[idx]
        # apply data augmentation
        # set geodesic distance attr to data
        if self.augment != None:
            subject_data_dict = self.augment.apply(subject_data_dict)
        data = torch_geometric.data.Data(
            x=torch.tensor(subject_data_dict['features'], dtype=torch.float),
            y=torch.tensor(subject_data_dict['labels'], dtype=torch.long),
            num_nodes=len(subject_data_dict['features']),)
        
        # add extra output levels to data
        if len(self.output_levels) != 0:
            labels_pooled = {7: data.y}
            for level in range(min(self.output_levels), 7)[::-1]:
                labels_pooled[level] = self.pool_layers[level](labels_pooled[level + 1])
            for level in self.output_levels:
                setattr(data, f"output_level{level}", labels_pooled[level])

            # add  distance maps if required
        if 'distances' in subject_data_dict.keys():
            #potentially here you could divide by 200
            setattr(data, "distance_map", 
            torch.tensor(subject_data_dict['distances'], dtype=torch.float))
            if len(self.output_levels) != 0:
                dists_pooled = {7: data.distance_map}
                for level in range(min(self.output_levels), 7)[::-1]:
                    dists_pooled[level] = self.pool_layers[level](dists_pooled[level + 1], center_pool=True)
                for level in self.output_levels:
                    setattr(data, f"output_level{level}_distance_map", dists_pooled[level])
        return data

    @property
    def lesional_idxs(self):
        """find ids of data entries with lesional examples"""
        if self._lesional_idxs is None:
            lesional_idxs = []
            for i, d in enumerate(self.data_list):
                if d['labels'].sum():
                    lesional_idxs.append(i)
            self._lesional_idxs = np.array(lesional_idxs)
            # self.log.info(f'lesional idxs: {self._lesional_idxs}')
        return self._lesional_idxs
