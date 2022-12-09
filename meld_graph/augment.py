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
    def __init__(self, params):
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
            
    def get_p_param(self, param):
        """check pvalue, set to zero if not found"""
        if param not in list(self.params.keys()):
            return 0
        else:
            return self.params[param]['p']
    
    def add_gaussian_noise(self,feat_tr):
        """ add a gaussian noise"""
        variance = np.random.uniform(0,0.1)
        feat_tr = feat_tr + np.random.normal(0.0, variance, size=feat_tr.shape)
        return feat_tr
    
    def add_gaussian_blur(self,feat_tr):
        """add gaussian blur function"""
        return feat_tr
    
    def add_brightness_scaling(self,feat_tr):
        """ scale brightness"""
        for c in range(feat_tr.shape[1]):
            multiplier = np.random.uniform(0.75, 1.25)
            feat_tr[:,c] *= multiplier
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
        for c in range(feat_tr.shape[1]):
            mn = feat_tr[:,c].mean()
            sd = feat_tr[:,c].std()
            gamma = np.random.uniform(0.7, 1.5)
            minm = feat_tr[:,c].min()
            rnge = feat_tr[:,c].max() - minm
            feat_tr[:,c] = np.power(((feat_tr[:,c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            feat_tr[:,c] = feat_tr[:,c] - feat_tr[:,c].mean()
            feat_tr[:,c] = feat_tr[:,c] / (feat_tr[:,c].std() + 1e-8) * sd
            feat_tr[:,c] = feat_tr[:,c] + mn
        return feat_tr
    
    def apply_indices(self,indices, tdd):
        
        # spin features
        tdd['features'] = tdd['features'][indices] 
        # spin lesions if exist
        if tdd['labels'].any()!= None:            
            tdd['labels'] = tdd['labels'][indices] 
        if 'distances' in tdd.keys():
            print(tdd['distances'].shape)
            tdd['distances'] = tdd['distances'][indices]
            
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
            
        #Gaussian blur - not implemented
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
        if np.random.rand() < self.get_p_param('gamma'):
            tdd['features'] = self.add_gamma_scale(tdd['features'])
        
        #inverted gamma intensity
        if np.random.rand() < self.get_p_param('gamma'):
            tdd['features'] = - self.add_gamma_scale( -tdd['features'])
            
        return tdd
    
   
       
