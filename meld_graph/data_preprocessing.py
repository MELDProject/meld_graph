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
import time

class Preprocess:
    params = {
        'scaling': None,
        'zscore': False
    }
    def __init__(self, cohort, site_codes=None, write_output_file=None, data_dir=BASE_PATH, params={}):
        """
        Load and preprocess data. 

        params:
            scaling: scale data between 0 and 1 using precomputed scaling params (TODO currently not implemented)
            zscore: z-score each feature for each subject (excluding medial wall)
        
        TODO : transform data if not gaussian 
        TODO : if not using combat features, correct for sulc freesurfer
        """
        self.cohort = cohort
        self.write_output_file = write_output_file
        self.data_dir = data_dir
        self.params.update(params)
        # private attributes for site_codes and subject_ids properties
        self._site_codes = site_codes
        self._subject_ids = None
        self.log = logging.getLogger(__name__)
        self._lobes = None
        self.initialise_distances()
        if self.params['zscore']:
            self.load_z_params()
         

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

    @property
    def lobes(self):
        if self._lobes is None:
            self._lobes = self.load_lobar_parcellation()
        return self._lobes
    

        
        
    def load_lobar_parcellation(self, lobe = 1):
        parc=nb.freesurfer.io.read_annot(os.path.join(self.data_dir,'fsaverage_sym','label','lh.lobes.annot'))[0]
        lobes = (parc==lobe).astype(int)
        return lobes


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
    
    def get_data_preprocessed(self, subject, features, lobes=False, lesion_bias=False):
        """
        Preprocess features data for a single subject depending on params.

        Args:
            lobes: if True, return lobes task as lesion values
            lesion_bias: if True, add lesion_bias value to lesional vertices. 
                NOTE: should not be used for final models, only for testing
        
        Returns:
            features_left, features_right, lesion_left, lesion_right
        """
        subj = MeldSubject(subject, cohort=self.cohort)  
        # load data & lesion
        vals_array_lh, lesion_lh = subj.load_feature_lesion_data(features, hemi='lh')
        vals_array_rh, lesion_rh = subj.load_feature_lesion_data(features, hemi='rh')
        # z-score data
        if self.params['zscore']:
            self.log.info(f"Z-scoring data for {subject}")
            vals_array_lh = self.zscore_data(vals_array_lh.T,features).T
            vals_array_rh = self.zscore_data(vals_array_rh.T,features).T
            
        if self.params['scaling'] is not None:
            self.log.info(f"Scaling data for {subject}")
            # all values excluding medial wall
            vals_array = np.array(np.hstack([vals_array_lh[self.cohort.cortex_mask].T, vals_array_rh[self.cohort.cortex_mask].T]))
        
            # correct for sulc freesurfer TODO only when not combat
    #         if '.on_lh.sulc.mgh' in features:
    #             index_sulc = features.index('.on_lh.sulc.mgh')
    #             vals_array[index_sulc] =  self.correct_sulc_freesurfer(vals_array[index_sulc])
        
            # scale data between 0 and 1  
            if self.params['scaling'] is not None:
                scaling_params_file = os.path.join(BASE_PATH, self.params['scaling'])
                vals_array = self.scale_data(vals_array, features, scaling_params_file )
            
            # transform data if feature not gaussian
            #TODO later
            
            # include medial wall back with 0 values
            vals_array_lh = np.zeros((NVERT, len(features)))
            vals_array_lh[self.cohort.cortex_mask] = vals_array[:, 0:sum(self.cohort.cortex_mask)].T
            vals_array_rh = np.zeros((NVERT, len(features)))
            vals_array_rh[self.cohort.cortex_mask] = vals_array[:, sum(self.cohort.cortex_mask) : sum(self.cohort.cortex_mask)*2].T
    
        # replace lesion data with lobes task if required
        if lobes:
            lesion_lh = self.lobes
            lesion_rh = self.lobes
        # add lesion bias
        if lesion_bias:
            self.log.info(f"WARNING: adding lesion bias of {lesion_bias} to {subject}")
            vals_array_lh[lesion_lh==1] += lesion_bias
            vals_array_rh[lesion_rh==1] += lesion_bias

        return vals_array_lh.T, vals_array_rh.T, lesion_lh, lesion_rh
    
    def load_z_params(self):
        import json
        with open('../data/feature_means.json', 'r') as fp:
            self.z_params=json.load( fp)
            
            
    def zscore_data(self, values,features):
        """zscore features using precalculated means and stds"""
        for fi,f_value in enumerate(values):
            if np.std(f_value)!=0:
                values[fi] = (f_value-self.z_params[features[fi]]['mean'])/self.z_params[features[fi]]['std']
        return values
        #old zscoring code, was problematic with synthetic lesions
#         std = values.std(axis=1, keepdims=True)
#         # for features with 0 std, set std to 1 to keep mean value (resulting in zscore 0)
#         std[std==0] = 1
#         return (values - values.mean(axis=1, keepdims=True)) / std
    
    

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

        
    def clockwiseangle_and_distance(self,point,origin):
        import math

        refvec = [0, 1]
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    def generate_synthetic_data(self,coords,n_features,
                                bias,radius=0.5,histo_type_seed=0,
                               proportion_features_abnormal = 0.2,
                                proportion_hemispheres_abnormal = 0.5,
                               jitter_factor=2,
                               features=None,
                               smooth_lesion=False):
        """coords - spherical coordinates
        n_features - number of input features
        bias  - mean of the difference in biases
        radius - mean size of lesion
        histo_type_seed - randomly generate different histologies.
        proportion_features_abnormal - proportion of features abnormal
        smooth_lesion - smooth edge of lesions"""
        
        
        #create a histological signature of -1,0,1 on which features are abnormal
        
        n_verts=len(coords)
        lesion = np.zeros(n_verts,dtype=bool)
        if features is None:
            features = np.random.normal(0,1,
                                    (n_features,n_verts))
            
        if np.random.random()<proportion_hemispheres_abnormal:
            features,lesion = self.add_lesion(features, coords,n_features,
                                bias,radius,histo_type_seed,
                               proportion_features_abnormal,jitter_factor,smooth_lesion=smooth_lesion)
        return features, lesion
    
    def clip_spherical_coords(self,coordinates):
        """make sure spherical coords in range"""
        coordinates[:,0]= np.clip(coordinates[:,0],-np.pi/2,np.pi/2)
        coordinates[:,1]= np.clip(coordinates[:,1],-np.pi,np.pi)
        return coordinates

    
    def initialise_distances(self,res=1000):
        """function to precalculate pairwise differences
        res - resolution, higher resolutions lead to slower synthetic generation
        lower resolutions give pixelated lesions"""
        from sklearn.metrics import pairwise_distances
        
        self.xnew = np.linspace(-np.pi/2,np.pi/2,res)
        self.ynew = np.linspace(-np.pi,np.pi,res*2)
        #calculate the shape of the grid
        self.gridshape=(2*res,res)
        #meshgrid, weird indexing required for the interpolator function
        self.grid_coords_grid = np.meshgrid(self.xnew,self.ynew,indexing='ij')
        self.grid_coords=np.vstack([self.grid_coords_grid[0].ravel(),self.grid_coords_grid[1].ravel()]).T

        self.origin = np.array([0,0])
        self.distances = pairwise_distances(self.origin.reshape(-1,1).T,
                                       self.grid_coords, metric='haversine')[0]
        return
        

    def create_lesion_mask(self,radius,cartesian_coords,return_smoothed=True):
        """create irregular polygon lesion mask"""
        import matplotlib.path as mpltPath
        #from sklearn.metrics import pairwise_distances
        from scipy import interpolate,ndimage
        import copy
        from meld_graph.resampling_meshes import spinning_coords
        from meld_classifier import mesh_tools as mt
        spun_coords = spinning_coords(cartesian_coords)
        spherical_coords = mt.spherical_np(spun_coords)[:,1:]
        spherical_coords[:,0] = spherical_coords[:,0]-np.pi/2
        spherical_coords = self.clip_spherical_coords(spherical_coords)

        #select a radius
        f_radius = np.clip(np.random.normal(radius,radius/2),0.05,2)
        n_points = np.random.choice(6)+4
        subset = self.grid_coords[self.distances<f_radius]
        #establish mask and mask coordinates
        x_mask = np.logical_and(self.grid_coords_grid[0]>-f_radius,self.grid_coords_grid[0]<f_radius)
        y_mask = np.logical_and(self.grid_coords_grid[1]>-f_radius,self.grid_coords_grid[1]<f_radius)
        grid_mask = np.logical_and(x_mask
            ,y_mask
            )
        mask_shape=(x_mask.any(axis=1).sum(),y_mask.any(axis=0).sum())
        masked_grid_coords = np.vstack([self.grid_coords_grid[0][grid_mask],
                      self.grid_coords_grid[1][grid_mask]]).T
        
        poly_i = np.random.choice(len(subset),n_points)
        polygon = subset[poly_i]
        polygon = np.array(sorted(polygon, key=lambda point: self.clockwiseangle_and_distance(point,self.origin)))
        path = mpltPath.Path(polygon)
        lesion = path.contains_points(masked_grid_coords)
        #lesion = path.contains_points(self.grid_coords)
        arr_lesion = lesion.reshape(mask_shape).astype(float)
        #arr_lesion = lesion.reshape(self.gridshape,order='f').astype(float)
        #interpolate to coordinates
        
        full_lesion = np.zeros(self.gridshape,dtype=float)
        full_lesion[grid_mask.T] = arr_lesion.T.ravel()
        f_near=interpolate.RegularGridInterpolator((self.xnew,self.ynew),
                                                   full_lesion.T,
                                                  method='nearest')
        interpolated_lesion=f_near(spherical_coords)
        #smoothed mask
        if return_smoothed:
            smoothed = ndimage.gaussian_filter(arr_lesion,10)
            full_lesion[grid_mask.T] = smoothed.T.ravel()

            f_lin=interpolate.RegularGridInterpolator((self.xnew,self.ynew),
                                                   full_lesion.T,
                                                  method='linear')
            #return grid_coords,smoothed
            interpolated_smoothed = f_lin(spherical_coords)
            return interpolated_lesion, interpolated_smoothed 
        else:
            return interpolated_lesion
    
    
    def sigmoid_dists(self,dists):
        m = dists==0
        z = 1/(1 + np.exp(-dists*5))
        z[m]=0
        return z
    
    def create_fingerprint(self,n_features, histo_type_seed, proportion_features_abnormal):
        """creates a vector of biases, seeded by the histological subtype integer.
        this is multiplied by the controlled bias term"""
        rng = np.random.default_rng(histo_type_seed) 
        feature_mask = rng.random(n_features)<proportion_features_abnormal
        histo_bias_multipliers = rng.random(size=n_features)
        histo_signature = rng.integers(low=0,high=2,size=n_features)*2-1
        fingerprint = feature_mask*histo_bias_multipliers*histo_signature
        return fingerprint
    
    def sample_fingerprint(self,fingerprint,jitter_factor):
        """use fingerprint as starting point for generating a slightly jittered individual fingerprint"""
        sampled_fingerprint=np.zeros_like(fingerprint)
        for fi,f in enumerate(fingerprint):
            if f!=0:
                sampled_fingerprint[fi] = np.random.normal(f,np.abs(f)/jitter_factor)
        return sampled_fingerprint
        
    def add_lesion(self, features,coords,n_features, bias, radius, histo_type_seed, 
                   proportion_features_abnormal,jitter_factor,smooth_lesion=False):
        """superimpose a synthetic lesion on input data 
       
       """
        #create lesion mask
        import time
        if smooth_lesion:
            lesion, smoothed_lesion = self.create_lesion_mask(radius,coords,return_smoothed=True)
        else:
            lesion = smoothed_lesion = self.create_lesion_mask(radius, coords, return_smoothed=False)
        lesion[~self.cohort.cortex_mask]=0
        smoothed_lesion[~self.cohort.cortex_mask]=0
        #bias is sampled from a normal dist so that some subjects are easier than others.
        sampled_bias = np.clip(np.random.normal(bias,bias/jitter_factor),0,100)
        #histo_signature - controls which features, how important and what sign
        fingerprint = self.create_fingerprint(n_features,histo_type_seed,proportion_features_abnormal)
        
        sampled_fingerprint = self.sample_fingerprint(fingerprint,jitter_factor)
        lesion_tiled = np.tile(smoothed_lesion.reshape(-1,1),
                                     n_features)
        synth_bias_features = (lesion_tiled*sampled_fingerprint*sampled_bias).T
        features= features + synth_bias_features
        
        return features,lesion
    
    
