from meld_classifier.paths import (
    BASE_PATH,
    NVERT,
)
import numpy as np
import nibabel as nb
import os
import logging
import json
import copy 
from meld_classifier.meld_cohort import MeldSubject


class Preprocess:
    """
    Load and preprocess data. 

    params:
        scaling: scale data between 0 and 1 using precomputed scaling params (TODO currently not implemented)
        zscore: z-score each feature for each subject (excluding medial wall)
    
    TODO : transform data if not gaussian 
    TODO : if not using combat features, correct for sulc freesurfer
    """
    params = {
        'scaling': None,
        'zscore': False
    }
    def __init__(self, cohort, site_codes=None, write_output_file=None,
    icospheres = None, data_dir=BASE_PATH, params={}):
        self.cohort = cohort
        self.write_output_file = write_output_file
        self.data_dir = data_dir
        self.params.update(params)
        # private attributes for site_codes and subject_ids properties
        self._site_codes = site_codes
        self._subject_ids = None
        self.log = logging.getLogger(__name__)
        self._lobes = None
        self._histo_encoder = None
        self.initialise_distances()
        self.icospheres = icospheres

        if self.params['zscore'] != False:
            self.load_z_params(self.params['zscore'])

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
    
    @property
    def histo_encoder(self):
        if self._histo_encoder is None:
            from sklearn.preprocessing import OneHotEncoder
            X=np.array(['FCD_1','FCD_2A','FCD_2B','FCD_3','no_lesion']).reshape(-1, 1)
            ohe = OneHotEncoder()
            ohe.fit(X)
            self._histo_encoder = ohe
        return self._histo_encoder

    def encode_histology(self, histology):
        if not histology in ['FCD_1', 'FCD_2A', 'FCD_2B', 'FCD_3', 'no_lesion']:
            return np.array([0,0,0,0,0]).reshape(1, -1)
        else:
            return self.histo_encoder.transform(np.array([histology]).reshape(-1, 1)).toarray()
        
    def get_data_preprocessed(self, subject, features, 
    lobes=False, lesion_bias=False,
    distance_maps = False, histology=False,
    combine_hemis=None, ):
        """
        Preprocess features data for a single subject depending on params.

        Args:
            lobes: if True, return lobes task as lesion values
            lesion_bias: if True, add lesion_bias value to lesional vertices. 
                NOTE: should not be used for final models, only for testing
                NOTE: this is an old flag, not in use in the current model anymore.
            distance_maps: read precalculated distance maps from subject.
                NOTE: this is an old flag, not in use anymore. 
                Distance are now calulated on the fly (because of lesion augmentation).
            combine_hemis: combine hemispheres to one sample by stacking.
            
        Returns:
            features_left, features_right, lesion_left, lesion_right
        """
        subj = MeldSubject(subject, cohort=self.cohort)
        subject_data = []
        # load data & lesion
        for hemi in ('lh','rh'):
            vals_array, lesion = subj.load_feature_lesion_data(features, hemi=hemi)
            subject_data_dict={}
        # z-score data
            if self.params['zscore']:
                if hemi =='lh':
                    self.log.info(f"Z-scoring data for {subject}")
                vals_array = self.zscore_data(vals_array.T,features).T
            if distance_maps:
                gdist = self.load_distances(subj,hemi)
                subject_data_dict['distances'] = gdist
            if self.params['scaling'] is not None:
                self.log.info(f"Scaling data for has been removed. REIMPLEMENT")
            if lobes:
                # replace lesion data with lobes task if required
                lesion = self.lobes
            # add lesion bias
            if lesion_bias:
                self.log.info(f"WARNING: adding lesion bias of {lesion_bias} to {subject}")
                vals_array[lesion==1] += lesion_bias
            if combine_hemis is not None:
                self.log.info(f"WARNING: combine_hemis is not implemented.")
            subject_data_dict['features'] = vals_array
            subject_data_dict['labels'] = lesion
            #add histology
            if histology:
                if subj.group == 'patient':
                    if lesion.sum()!=0:
                        histology =  subj.get_demographic_features('Histology')
                    else:
                        histology = 'no_lesion'
                else:
                    histology = 'no_lesion'
                subject_data_dict['histology'] = self.encode_histology(histology) 
                
            subject_data.append(subject_data_dict)
        return subject_data
    
    def load_distances(self,subj,hemi='lh'):
        """load geodesic distance from lesion or 300s"""
        if (not subj.is_patient) or (subj.get_lesion_hemisphere() != hemi):
            gdist = np.ones(NVERT, dtype=np.float32)*300

        else:
            gdist = subj.load_feature_values('.on_lh.boundary_zone.mgh',
                    hemi=hemi)
            # threshold to range 0,300
            gdist = np.clip(gdist, 0, 300)
        return gdist
    
    def load_z_params(self, file='../data/feature_means.json'):
        import json
        with open(file, 'r') as fp:
            self.z_params=json.load( fp)
              
    def zscore_data(self, values,features):
        """zscore features using precalculated means and stds"""
        for fi,f_value in enumerate(values):
            if np.std(f_value)!=0:
                values[fi] = (f_value-self.z_params[features[fi]]['mean'])/self.z_params[features[fi]]['std']
        return values

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

    def generate_synthetic_data(self,coords,n_features,synth_params,
                                histo_type_seed=0,
                               features=None,
                               ):
        """coords - spherical coordinates
        n_features - number of input features
        bias  - mean of the difference in biases
        radius - mean size of lesion
        histo_type_seed - randomly generate different histologies.
        proportion_features_abnormal - proportion of features abnormal
        smooth_lesion - smooth edge of lesions"""
        #create a histological signature of -1,0,1 on which features are abnormal
        n_verts=len(coords)
        lesion = np.zeros(n_verts,dtype=int)
        if features is None:
            features = np.random.normal(0,1,
                                    (n_verts,n_features))
        
        if np.random.random()<synth_params['proportion_hemispheres_lesional']:
            synth_dict = self.add_lesion(features, coords,n_features,
                                synth_params,histo_type_seed,
                               )
        else: 
            synth_dict = {'features':features,'labels':lesion,
            }
            
        return synth_dict
    
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
        #make sure there are enough lesional vertices
        lesional_verts = -1
        while lesional_verts < 1:
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
            lesional_verts = interpolated_lesion.sum()
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
        
    def add_lesion(self, features,coords,n_features, synth_params, histo_type_seed, 
                   ):
        """superimpose a synthetic lesion on input data 
       
       """
        #create lesion mask
        if synth_params['smooth_lesion']:
            lesion, smoothed_lesion = self.create_lesion_mask(synth_params['radius'],coords,
                  return_smoothed=True)
        else:
            lesion = smoothed_lesion = self.create_lesion_mask(synth_params['radius'], coords, return_smoothed=False)
        lesion[~self.cohort.cortex_mask]=0
        smoothed_lesion[~self.cohort.cortex_mask]=0
        #bias is sampled from a normal dist so that some subjects are easier than others.
        sampled_bias = np.clip(np.random.normal(synth_params['bias'],
                    synth_params['bias']/synth_params['jitter_factor']),0,100)
        #histo_signature - controls which features, how important and what sign
        fingerprint = self.create_fingerprint(n_features,histo_type_seed,synth_params['proportion_features_abnormal'])

        sampled_fingerprint = self.sample_fingerprint(fingerprint,synth_params['jitter_factor'])
        lesion_tiled = np.tile(smoothed_lesion.reshape(-1,1),
                                     n_features)
        synth_bias_features = (lesion_tiled*sampled_fingerprint*sampled_bias)
        #apply synthetic lesion only on non-null feature
        apply = np.array([1 if feature.any() != 0 else 0 for feature in features.T])
        features= features + synth_bias_features*apply
        synth_dict = {'features' : features.astype('float32'),
                      'labels' : lesion.astype('int32')
            }
                
        return synth_dict
    