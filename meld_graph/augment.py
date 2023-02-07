#augment class
import os
import numpy as np
import nibabel as nb
import copy
import time
from scipy import sparse 
import meld_classifier.mesh_tools as mt
import torch
from math import pi 
import logging
from meld_graph.paths import (
    SCRIPTS_DIR,)

from meld_graph.models import  HexSmooth


class Transform():
    """Class transform paramaters"""
    def __init__(self, params_transform):
        self.p = params_transform['p']
        self.indices = np.load(os.path.join(SCRIPTS_DIR,params_transform['file']))
        self.indices = self.indices.astype('int') 
    
    def apply_transform_old(self, feats, lesions=None):
        # select random transformation parameter
        transf = np.random.randint(0,len(self.lambdas))
        # spin lesions if exist
        if lesions.any()!= None:            
            lesions_transf = self.lambdas[transf,:,0]*lesions[self.indices[transf,:,0]] + self.lambdas[transf,:,1]*lesions[self.indices[transf,:,1]] + self.lambdas[transf,:,2]*lesions[self.indices[transf,:,2]]   
            lesions_transf = np.round(lesions_transf)
        # spin features
        n_feat = len(feats.T)
        lambdas = np.tile(self.lambdas[:,:,:,np.newaxis], n_feat )
        feats_transf = lambdas[transf,:,0]*feats[self.indices[transf,:,0]] + lambdas[transf,:,1]*feats[self.indices[transf,:,1]] + lambdas[transf,:,2]*feats[self.indices[transf,:,2]]        
        feats_transf_clean=np.zeros(feats_transf.shape)
        for i in range(0,n_feat):
            feats_transf_clean[:,i]=np.clip(feats_transf[:,i], np.percentile(feats_transf[:,i], 0.01),np.percentile(feats_transf[:,i], 99.9))  
        return feats_transf_clean, lesions_transf
    
    #fastest version
    def apply_transform(self, feats, lesions=None):
        #we don't need to use this, even though it's correct, nearest is much faster.
        # select random transformation parameter
        transf = np.random.randint(0,len(self.lambdas))
        #initiate lambdas and indices to speed up
        indices=copy.deepcopy(self.indices[transf])
        i0=indices[:,0]
        i1=indices[:,1]
        i2=indices[:,2]
        lambdas=copy.deepcopy(self.lambdas[transf])
        l0=lambdas[:,0]
        l1=lambdas[:,1]
        l2=lambdas[:,2]
        # spin lesions if exist
        if lesions.any()!= None:            
            lesions_transf = l0*lesions[i0] + l1*lesions[i1] + l2*lesions[i2]   
            lesions_transf = np.round(lesions_transf)
        # spin features
        n_feat = len(feats.T)
        l0 = np.tile(l0[:,np.newaxis], n_feat)
        l1 = np.tile(l1[:,np.newaxis], n_feat)
        l2 = np.tile(l2[:,np.newaxis], n_feat)
        feats_transf = l0*feats[i0] + l1*feats[i1] + l2*feats[i2]        
        feats_transf_clean=np.zeros(feats_transf.shape)
        feats_transf_clean=np.clip(feats_transf, np.percentile(feats_transf, 0.01),np.percentile(feats_transf, 99.9)) 
        return feats_transf_clean, lesions_transf
    
    def apply_transform_nearest(self, feats, lesions=None):
        # select random transformation parameter
        transf = np.random.randint(0,len(self.indices))
        #initiate lambdas and indices to speed up
        indices=copy.deepcopy(self.indices[transf])
        # spin lesions if exist
        if lesions.any()!= None:            
            lesions_transf = lesions[indices] 
        # spin features
        feats_transf = feats[indices] 
        return feats_transf, lesions_transf
    
    def get_indices(self):
        transf = np.random.randint(0,len(self.indices))
        #initiate lambdas and indices to speed up
        indices = copy.deepcopy(self.indices[transf])
        return indices

class Augment():
    """Class to augment data"""
    def __init__(self, params,graph_tools):
        """Augment class
        params - dictionary containing augmentation method, file, and probability of apply transformation (p)
        following guidance from nnUNET
        transformations in the following order:
        Rotation & scaling - p=0.2
        Gaussian noise - p=0.15, mu=0,std = U(0,0.1)
        Gaussian blur - p=0.2 if applied, p=0.5 per modality if triggered, width in voxels is U(0.5,1.5)
        Brightness - *x U(0.7,1.3) p =0.15 to all modalities
        Contrast - *x U(0.65,1.5) p=0.15, clipped to original range to all modalidies
        Low resolution - p=0.25, if applied p=0.5 per modality, Down sampled U(1,2) nearest neighbour
        Gamma augmentation - invert p=0.1, Gamma, non-invert p=0.3
        Mirroring - flipping p=0.5
        """ 
        self.log = logging.getLogger(__name__)
        self.params=params
        self.transform_types= set(self.params)
        if 'spinning' in self.transform_types:
            self.spinning = Transform(self.params['spinning'])
        else:
            self.spinning = None
        if 'warping' in self.transform_types:
            self.warping = Transform(self.params['warping'])
        else:
            self.warping = None
        if 'flipping' in self.transform_types:
            self.flipping = Transform(self.params['flipping'])
        else:
            self.flipping = None

        self.gt = graph_tools
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #need to load neighbours
        #self.smooth_step = HexSmooth(neighbours = neighbours)
            
    def get_p_param(self, param):
        """check pvalue, set to zero if not found"""
        if param not in list(self.params.keys()):
            return 0
        else:
            return self.params[param]['p']
    
    def add_gaussian_noise(self,feat_tr):
        """ add a gaussian noise"""
        variance = np.random.uniform(0,0.1)
        feat_tr = feat_tr + np.random.normal(0.0, variance, size=feat_tr.shape,
                                             ).astype(np.float16)
        #feat_tr = feat_tr + variance * np.random.randn(*feat_tr.shape)
        return feat_tr
    
    def add_gaussian_blur(self,feat_tr):
        """add gaussian blur function"""
        #n_iterations = np.random.choice(10)
        #for iteration in np.arange(n_iterations):
        #    feat_tr = self.smooth_step(feat_tr)
        return feat_tr

    
    
    
    
    def add_brightness_scaling(self,feat_tr):
        multipliers = np.random.uniform(0.75, 1.25, size=feat_tr.shape[1])
        feat_tr *= multipliers[None,:]
        return feat_tr
    
    def adjust_contrast(self,feat_tr):
        """adjust contrast"""
        for c in range(feat_tr.shape[1]):
            factor = np.random.uniform(0.65,1.5)
            mn = feat_tr[:,c].mean()
            minm = feat_tr[:,c].min()
            maxm = feat_tr[:,c].max()
            feat_tr[:,c] = (feat_tr[:,c] - mn) * factor + mn            
            feat_tr[:,c][feat_tr[:,c] < minm] = minm
            feat_tr[:,c][feat_tr[:,c] > maxm] = maxm
        return feat_tr
    
    def add_low_res(self,feat_tr):
        """add low resolution version"""
        return feat_tr
    
    def add_gamma_scale(self,feat_tr):
        """ add gamma scaling"""
        epsilon=1e-7
        mn = feat_tr.mean(axis=0)
        sd = feat_tr.std(axis=0)
        minm = feat_tr.min(axis=0)
        rnge = feat_tr.max(axis=0) - minm
        gamma = np.random.uniform(0.7, 1.5, size=feat_tr.shape[1]).astype(np.float16)
        feat_tr = np.power(((feat_tr - minm) / (rnge + epsilon)), gamma) * (rnge + epsilon) + minm
        feat_tr = (feat_tr - mn) / (sd + epsilon)
        return feat_tr
    
    def extend_lesion(self, lesions, distances):
        """ DEFUNCT not using this any more"""
        if (not (lesions==1).any()) or ((distances==200).all()):
            return lesions, distances
        extension = np.random.choice(np.linspace(1, 20, 20))
        #update distances
        distances_extend = np.clip(distances-extension,a_min=0, a_max=None)
        # extend lesions
        lesions_extend = (distances_extend<=0)
        return lesions_extend, distances_extend

    def augment_lesion(self, tdd, noise_std=0.5):
        # modify lesion using low frequency noise
       
        # get geodesic distance (negative inside lesion, positive outside)
        # normalise by minimum values
        new_dist = tdd['distances']
        new_dist_norm = new_dist / np.abs(new_dist.min())
        # create low frequencies noise on low res icosphere 2
        n_vert_low = len(self.gt.icospheres.icospheres[2]['coords'])
        noise = np.random.normal(0,noise_std,n_vert_low)
        #upsample noise to high res
        for level in range(2, 7):
            unpool_ind = self.gt.unpool(level=level+1)
            noise_upsampled = unpool_ind(torch.from_numpy(noise.reshape(-1,1)), device = self.device)
            noise_upsampled = noise_upsampled.detach().cpu().numpy().ravel()
            noise = noise_upsampled.copy()
        #add noise to distance normalised
        new_mask = (new_dist_norm + noise_upsampled)<=0
        tdd['labels'] = new_mask
        return tdd

    def recompute_distance_and_smoothed(self,tdd):
        """recompute distances from augmented lesion masks"""
        tdd['distances'] = self.gt.fast_geodesics(tdd['labels'])
        tdd['smooth_labels'] = self.gt.smoothing(tdd['labels'],iteration=10)
        
        return


    def apply_indices(self,indices, tdd):
        # spin features
        tdd['features'] = tdd['features'][indices] 
        # spin lesions if exist
        if (tdd['labels']==1).any():            
            tdd['labels'] = tdd['labels'][indices] 
        for field in tdd.keys():
            if field=='labels':
                if (tdd['labels']==1).any():
                    tdd['labels'] = tdd['labels'][indices] 
            else:
                tdd[field] = tdd[field]
        return tdd
       
    def apply(self, subject_data_dict):
        #create a transformed data dict
        tdd = subject_data_dict.copy()
        #spinning   
        mesh_transform = False
        indices = np.arange(tdd['features'].shape[0],dtype=int)
        #stack the transformations into a single indexing step
        if np.random.rand() < self.get_p_param('spinning'):
            mesh_transform = True
            indices = indices[self.spinning.get_indices()]
            #feat_tr, lesions_tr= self.spinning.apply_transform(feat_tr, lesions_tr)
        #warping
        if np.random.rand() < self.get_p_param('warping'):
            mesh_transform = True
            indices = indices[self.warping.get_indices()]
            #feat_tr, lesions_tr= self.warping.apply_transform(feat_tr, lesions_tr)
            
        if np.random.rand() < self.get_p_param('flipping'):
            mesh_transform = True
            indices = indices[self.flipping.get_indices()]
            #feat_tr, lesions_tr= self.flipping.apply_transform(feat_tr, lesions_tr)
        
        #apply just once
        if mesh_transform:
            tdd = self.apply_indices(indices,
            tdd)
        #Gaussian noise
        if np.random.rand() < self.get_p_param('noise'):
            tdd['features'] = self.add_gaussian_noise(tdd['features'])
        
        #Gaussian blur 
        if np.random.rand() < self.get_p_param('blur'):
            tdd['features']= self.add_gaussian_blur(tdd['features'])
        #Brightness scaling
        if np.random.rand() < self.get_p_param('brightness'):
            tdd['features']= self.add_brightness_scaling(tdd['features'])
        
        #adjust contrast
        if np.random.rand() < self.get_p_param('contrast'):
            tdd['features']= self.add_brightness_scaling(tdd['features'])
        #low res - not implemented
        if np.random.rand() < self.get_p_param('low_res'):
            tdd['features']= self.add_low_res(tdd['features'])

        #gamma intensity
        if np.random.rand() < self.get_p_param('gamma')/2:
            tdd['features'] = self.add_gamma_scale(tdd['features'])
        #inverted gamma intensity
        if np.random.rand() < self.get_p_param('gamma')/2:
            tdd['features'] = - self.add_gamma_scale( -tdd['features'])
        
        #randomly augment lesion using distances and noise
        if (tdd['labels']==1).any():
            if np.random.rand() < self.get_p_param('augment_lesion'):
                tdd = self.augment_lesion(tdd)
                self.recompute_distance_and_smoothed(tdd)
        
       # if (np.random.rand() < self.get_p_param('extend_lesion')) & ('distances' in tdd.keys()):
       #     tdd['labels'], tdd['distances']=self.extend_lesion(tdd['labels'], tdd['distances'])
 
        return tdd
    
   
       
