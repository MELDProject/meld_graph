import torch_geometric.data
from meld_graph.data_preprocessing import Preprocess
from meld_graph.icospheres import IcoSpheres
from meld_graph.models import HexPool
from meld_graph.augment import Augment
import numpy as np
import torch
import logging
from meld_graph.graph_tools import GraphTools

import torch


        
class Oversampler(torch.utils.data.Sampler):
    """
    NNUnet-like oversampling.
    33% lesional, 66% random.

    Shuffles data after sampling.
    """
    def __init__(self, data_source):
        self.log = logging.getLogger(__name__)
        self.data_source = data_source
        self._num_samples = None
        self._num_per_histos = None

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        if self._num_samples is None:
            # find the maximum number of sample of histology
            return int(np.max(self.num_per_histos))
        return self._num_samples
    
    @property
    def num_per_histos(self):
        if self._num_per_histos is None:
            # find the number of samples for each histology 
            return np.array([len(x) for x in self.data_source.histo_idxs[0:3]]).astype('int64')
        return self._num_per_histos
    

    def get_sampling_idxs(self):
        """
        Return list of len num_samples, containing data_source idxs to sample.
        Call once per epoch (when restarting iterator).
        """
        # get the number of random samples to match the number of histo samples (max number sample histologies * number of histologies)
        n_non = self.num_samples*(self.num_per_histos>0).sum()
        # randomly select n_non samples among all samples 
        non_ids = torch.randperm(len(self.data_source), dtype=torch.int64)[:n_non]
        # randomly select the same number of sample for each histologies than one that have the most of sample        n_0 = np.random.choice(self.data_source.histo_idxs[0], size=self.num_samples, replace=True)
        n_0 = np.random.choice(self.data_source.histo_idxs[0], size=self.num_samples, replace=True) # FCD 1
        n_1 = np.random.choice(self.data_source.histo_idxs[1], size=self.num_samples, replace=True) # FCD 2A
        n_2 = np.random.choice(self.data_source.histo_idxs[2], size=self.num_samples, replace=True) # FCD 2B
        # stack the histological samples with the random sample 
        ids_to_choose = torch.hstack(
            [torch.from_numpy(n_0), torch.from_numpy(n_1), torch.from_numpy(n_2), non_ids]
        )
        #return list of samples permute
        return ids_to_choose[torch.randperm(len(ids_to_choose), dtype=torch.int64)]

    def __iter__(self):
        ids_to_choose = self.get_sampling_idxs()
        return iter(ids_to_choose.tolist())

    def __len__(self):
        return self.num_samples

class GraphDataset(torch_geometric.data.Dataset):
    """
    GraphDataset containing hemisphere-level data.

    Always returns data (x), labels (y), and distances (distance_map). 
    Distances are clipped to 300, and for controls, distances are set to maximum value of 300.

    Depending on params, lesions will be simulated or real. 

    Args:
        subject_ids (list): subjects to iterate over.
        cohort (MeldCohort): cohort defining how to read data.
        params (dict): data_parameters. See example_experiment_config.py for options.
        mode (str): when train, augment data.
        output_levels (list): icosphere levels for which y should be returned as well. Used for deep supervision.
            will be available as self.get().output_level<level>.
            Distance maps will be available as self.get().output_level<level>_distance_map.
        distance_mask_medial_wall (bool): mask of medial wall in distance maps to 300.
        get_histology (bool): return the histology as a class
    """
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
        distance_mask_medial_wall=True,
    ):
       
        super().__init__(None, transform, pre_transform, pre_filter)
        self.log = logging.getLogger(__name__)
        self.params = params
        self.subject_ids = subject_ids
        self.cohort = cohort
        self.mode = mode
        self.output_levels = sorted(output_levels)
        self.icospheres = IcoSpheres()
        self.gt = GraphTools(self.icospheres, cohort=self.cohort, distance_mask_medial_wall=distance_mask_medial_wall)
        self.augment = None
        self.index_overwrite = None
        if (self.mode == "train") & (self.params["augment_data"] != None):
            self.augment = Augment(self.params["augment_data"], self.gt)
            # create mask to overwrite the lesion mask if given as input feature after augmentation 
        if ".on_lh.lesion.mgh" in self.params["features"]:
            self.index_overwrite = np.where(np.array(self.params["features"])==".on_lh.lesion.mgh")[0][0]   
        if len(self.output_levels) != 0:
            self.pool_layers = {
                level: HexPool(self.icospheres.get_downsample(target_level=level))
                for level in range(min(self.output_levels), 7)[::-1]
            }
        self._lesional_idxs = None
        self.use_histology = self.params.get('use_histology', False)
        self._histo_idxs = None
        self._num_histos = None


        # preload data in memory, with all preprocessing done
        self.data_list = []
        self.prep = Preprocess(
            cohort=self.cohort, params=self.params["preprocessing_parameters"],
            icospheres = self.icospheres
        )
        self.log.info(f"Loading and preprocessing {mode} data")
        self.log.debug(f"Combine hemis {self.params['combine_hemis']}")
        
        if self.params["synthetic_data"]["run_synthetic"]:     
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
            # undersample subject ids to get controlled number
            n_subs_before = len(self.subject_ids)
            if self.n_subs_split<=len(self.subject_ids):
                self.subject_ids = np.random.choice(self.subject_ids,self.n_subs_split)
                self.subject_samples = np.arange(len(self.subject_ids))
            elif self.n_subs_split>len(self.subject_ids):
                # if wanting multiple samples of same subjects
                self.subject_samples = np.sort(np.random.choice(np.arange(len(self.subject_ids)),
                                                        self.n_subs_split))
        
            self.log.info(f"WARNING: Simulating {len(self.subject_samples)} subjects using {n_subs_before} controls")                                       
    
        for s_i,subj_id in enumerate(self.subject_ids):
            # load in (control) data
            # features are appended to list in order: left, right
            subject_data_list = self.prep.get_data_preprocessed(subject=subj_id, 
                                        features=params['features'], 
                                        lobes = params['lobes'], lesion_bias=False,
                                        distance_maps=False, histology=self.use_histology,
                                        combine_hemis = self.params['combine_hemis'])
            
            # add lesion if simulating synthetic data
            if self.params['synthetic_data']['run_synthetic']:
                for duplicate in np.arange(np.sum(self.subject_samples==s_i)):
                        synth_sub_data_list = self.synthetic_lesion(subject_data_list)
                        # computing dists and smoothed labels
                        synth_sub_data_list= self.add_smooth_label_and_dists(synth_sub_data_list)
                        self.data_list.extend(synth_sub_data_list)
            else:
                # add dists and smoothed labels
                subject_data_list = self.add_smooth_label_and_dists(subject_data_list)
                self.data_list.extend(subject_data_list)

        # dataset has weird properties. subject_ids needs to be the right length, matching the data length
        if self.params['synthetic_data']['run_synthetic']:
            if self.n_subs_split>len(self.subject_ids):
                self.subject_ids = np.array(self.subject_ids)[self.subject_samples]
        return

    @classmethod
    def from_experiment(cls, experiment, mode):
        """
        Initialise GraphDataset from experiment.
        """
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
            output_levels=experiment.network_parameters["training_parameters"].get("deep_supervision", {}).get("levels", []),
            distance_mask_medial_wall=experiment.data_parameters.get('distance_mask_medial_wall', False),
        )
    
    def add_smooth_label_and_dists(self,subject_data_list):
        """Compute a smoothed label and distance map.
        
        Updates subject_data_list with "smooth_labels" and "distances".
        """
        for sdl in subject_data_list:
            if (sdl['labels']==1).any():
                if self.params['smooth_labels']:
                    sdl['smooth_labels'] = self.gt.smoothing(sdl['labels'],iteration=10).astype(np.float32)
                sdl['distances'] = self.gt.fast_geodesics(sdl['labels']).astype(np.float32)
            else:
                sdl['distances'] = np.ones(len(sdl['labels']),dtype=np.float32)*300
                if self.params['smooth_labels']:
                    sdl['smooth_labels'] = np.zeros(len(sdl['labels']),dtype=np.float32)
        return subject_data_list
    
    def synthetic_lesion(self, subject_data_list=[{'features':None},
                                                  {'features':None}]):
        """Add synthetic lesion to input features for both hemis"""
        synth_dicts=[]
        for si,sdl in enumerate(subject_data_list):
            # controls the proportion of examples with lesions.
            subtype=np.random.choice(self.params['synthetic_data']['n_subtypes'])
            synth_dict = self.prep.generate_synthetic_data(self.icospheres.icospheres[7]['coords'],
                                                              len(self.params['features']),
                                                             histo_type_seed=subtype,
                                                             synth_params = self.params['synthetic_data'],
                                                    features=sdl['features'],
                                                    )
            synth_dicts.append(synth_dict)
        return synth_dicts

    def len(self):
        # every subject will be shown twice per epoch
        return len(self.data_list)

    def get(self, idx):
        """
        Return single data point.

        Returns data will have attributes x, y, distance_map, output_level<level>, output_level<level>_distance_map.
        """
        subject_data_dict = self.data_list[idx]
        # apply data augmentation
        if self.augment != None:
            subject_data_dict = self.augment.apply(subject_data_dict)
        
        # overwrite lesion mask by labels if given as input features
        if self.index_overwrite != None:
            subject_data_dict['features'][:,self.index_overwrite]=subject_data_dict['labels']

        if self.params['smooth_labels'] and self.augment!=None:
            data = torch_geometric.data.Data(
        x=torch.tensor(subject_data_dict['features'], dtype=torch.float32),
        y=torch.tensor(subject_data_dict['smooth_labels'], dtype=torch.float32),
        num_nodes=len(subject_data_dict['features']),)
        else:
            data = torch_geometric.data.Data(
        x=torch.tensor(subject_data_dict['features'], dtype=torch.float32),
        y=torch.tensor(subject_data_dict['labels'], dtype=torch.int64),
        num_nodes=len(subject_data_dict['features']),)

        # add histology class to data
        if self.use_histology:
            setattr(data, f"histology_class", torch.tensor(subject_data_dict['histology'], dtype=torch.float32))

        # add extra output levels to data
        if len(self.output_levels) != 0:
            labels_pooled = {7: data.y}
            for level in range(min(self.output_levels), 7)[::-1]:
                labels_pooled[level] = self.pool_layers[level](labels_pooled[level + 1])
            for level in self.output_levels:
                setattr(data, f"output_level{level}", labels_pooled[level])

        # add  distance maps
        # clip distances
        setattr(data, "distance_map", torch.tensor(np.clip(subject_data_dict['distances'], 0, 300), dtype=torch.float32))
        if len(self.output_levels) != 0:
            dists_pooled = {7: data.distance_map}
            for level in range(min(self.output_levels), 7)[::-1]:
                dists_pooled[level] = self.pool_layers[level](dists_pooled[level + 1], center_pool=True)
            for level in self.output_levels:
                setattr(data, f"output_level{level}_distance_map", torch.clip(dists_pooled[level],0,300))
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
        return self._lesional_idxs
    
    @property
    def histo_idxs(self):
        """find ids of data entries with each histology examples"""
        if self._histo_idxs is None:
            histo_0_idxs = []
            histo_1_idxs = []
            histo_2_idxs = []
            histo_3_idxs = []
            for i, d in enumerate(self.data_list):
                if d['histology'][0][0]==1:
                    histo_0_idxs.append(i)
                elif d['histology'][0][1]==1:
                    histo_1_idxs.append(i)
                elif d['histology'][0][2]==1:
                    histo_2_idxs.append(i)
                elif d['histology'][0][3]==1:
                    histo_3_idxs.append(i)
                else:
                    pass
            self._histo_idxs = np.array([np.array(histo_0_idxs), np.array(histo_1_idxs), np.array(histo_2_idxs), np.array(histo_3_idxs)])
        return self._histo_idxs
    
    
