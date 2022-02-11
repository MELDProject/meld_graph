from meld_classifier.paths import (
    BASE_PATH,
    NVERT,
)
import pandas as pd
import numpy as np
import nibabel as nb
import os
import h5py
import glob
import logging
import random
import json
import copy 
from meld_classifier.meld_cohort import MeldCohort, MeldSubject


class Preprocess:
    def __init__(self, cohort, site_codes=None, write_output_file=None, data_dir=BASE_PATH):
        self.cohort = cohort
        self.write_output_file = write_output_file
        self.data_dir = data_dir
        # private attributes for site_codes and subject_ids properties
        self._site_codes = site_codes
        self._subject_ids = None
        self.log = logging.getLogger(__name__)
        
    @property
    def site_codes(self):
        if self._site_codes is None:
            self._site_codes = self.cohort.get_sites()
        return self._site_codes
    
    @property
    def subject_ids(self):
        if self._subject_ids is None:
            # filter subject ids based on site codes
            self._subject_ids = self.cohort.get_subject_ids(site_codes=self.site_codes, lesional_only=False)
        return self._subject_ids

        
    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def save_cohort_features(self, feature_name, features, subject_ids, hemis=["lh", "rh"]):
        assert len(features) == len(subject_ids)
        for s, subject in enumerate(subject_ids):
            subj = MeldSubject(subject, cohort=self.cohort)
            subj.write_feature_values(feature_name, features[s], hemis=hemis, hdf5_file_root=self.write_hdf5_file_root)

    def correct_sulc_freesurfer(self, vals):
        """this function normalized sulcul feature in cm when values are in mm (depending on Freesurfer version used)"""
        if np.mean(vals, axis=0) > 0.2:
            vals = vals / 10
        else:
            pass
        return vals

    
    def get_data_preprocessed(self, subject, features, params):
        ''' This function preprocessed features data for a single subject$
        preprocess: 
        1) get data and lesion
        2) scale data between 0 and 1
        3) TODO : transform data if not gaussian 
        '''
        
        subj = MeldSubject(subject, cohort=self.cohort)  
        #load data & lesion
        vals_array_lh, lesion_lh = subj.load_feature_lesion_data(features, hemi='lh')
        vals_array_rh, lesion_rh = subj.load_feature_lesion_data(features, hemi='rh')
        vals_array = np.array(np.hstack([vals_array_lh[self.cohort.cortex_mask].T, vals_array_rh[self.cohort.cortex_mask].T]))
        
        #correct for sulc freesurfer
        if '.on_lh.sulc.mgh' in features:
            index_sulc = features.index('.on_lh.sulc.mgh')
            vals_array[index_sulc] =  self.correct_sulc_freesurfer(vals_array[index_sulc])
        
        #if flag 'sclaling' scale data between 0 and 1  
        if params['scaling'] != None:
            scaling_params_file = os.path.join(BASE_PATH,params['scaling'])
            preprocessed_data = self.scale_data(vals_array, features, scaling_params_file )
        else:
            preprocessed_data = copy.deepcopy(vals_array)
       
        #transform data if feature not gaussian
        #TODO later
        
        #include medial wall back with 0 values
        features_lh = np.zeros((len(features), NVERT))
        features_lh[:, self.cohort.cortex_mask] = preprocessed_data[:, 0:sum(self.cohort.cortex_mask)]
        features_rh = np.zeros((len(features), NVERT))
        features_rh[:, self.cohort.cortex_mask] = preprocessed_data[:, sum(self.cohort.cortex_mask) : sum(self.cohort.cortex_mask)*2]
        
        return features_lh, features_rh, lesion_lh, lesion_rh
    
    def scale_data(self, matrix, features, file_name):
        """scale data features between 0 and 1"""
        self.log.info(f"Scale data using file {file_name}")
        file = os.path.join(BASE_PATH, file_name)
        with open(file, "r") as f:
            params_norm = json.loads(f.read())
        data = copy.deepcopy(matrix)
        #scale and clip data
        for f, feature in enumerate(features):
            min_val = float(params_norm[feature]["min"])
            max_val = float(params_norm[feature]["max"])
            data[f] = np.clip(data[f], min_val, max_val)
            data[f] = (data[f]-min_val)/(max_val - min_val)
        return data        
        
    def compute_scaling_parameters(self, feature):
        """get mean and std of all brain for the given cohort and save parameters"""
        print(f"Compute scaling values for feature {feature}")
        # Give warning if list of subjects empty
        if len(self.subject_ids) == 0:
            print("WARNING: there is no subject in this cohort")
        vals_array = []
        included_subj = []
        for id_sub in self.subject_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[self.cohort.cortex_mask], vals_rh[self.cohort.cortex_mask]]))
                if feature == ".on_lh.sulc.mgh":
                    vals =  self.correct_sulc_freesurfer(vals)
                vals_array.append(vals)
                included_subj.append(id_sub)
            else:
                pass
        self.log.info("Compute min and max from {} subjects".format(len(included_subj), feature))
        # get min and max percentile
        vals_array = np.array(vals_array)
        min_val=np.percentile(vals_array.flatten(), 1, axis=0)
        max_val= np.percentile(vals_array.flatten(), 99, axis=0)  
        self.log.info(f'min value = {min_val}')
        self.log.info(f'max value = {max_val}')
        # save in json
        data = {}
        data["{}".format(feature)] = {
            "min": str(min_val),
            "max": str(max_val),
        }
        # create or re-write json file
        file = os.path.join(self.data_dir, self.write_output_file)
        if os.path.isfile(file):
            # open json file and get dictionary
            with open(file, "r") as f:
                x = json.loads(f.read())
            # update dictionary with new dataset version
            x.update(data)
        else:
            x = data
        # save dictionary in json file
        with open(file, "w") as outfile:
            json.dump(x, outfile, indent=4)
        print(f"parameters saved in {file}")
