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
        self.lambdas, self.indices = np.load(os.path.join(SCRIPTS_DIR,params_transform['file']))
        self.indices = self.indices.astype('int') 
        
    def apply_transform(self, feats, lesions=None):
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
#         for i in range(0,n_feat):
#             feats_transf_clean[:,i]=np.clip(feats_transf[:,i], np.percentile(feats_transf[:,i], 0.01),np.percentile(feats_transf[:,i], 99.9))  
        return feats_transf_clean, lesions_transf


class Augment():
    """Class to augment data"""
    def __init__(self, params):
        """Augment class
        params - dictionary containing augmentation method, file, and probability of apply transformation (p)
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
       
    def apply(self, features, lesions=None):
        feat_tr = features
        lesions_tr = lesions
        #spinning
        if self.spinning != None:
            random_p = np.random.rand()
            self.log.debug(f'random probability for spinning : {random_p}')
            if random_p < self.spinning.p:
                self.log.debug('apply spinning')
                feat_tr, lesions_tr= self.spinning.apply_transform(feat_tr, lesions_tr)
        #flipping
        if self.flipping != None:
            random_p = np.random.rand()
            self.log.debug(f'random probability for flipping : {random_p}')
            if random_p < self.flipping.p:
                self.log.debug('apply flipping')
                feat_tr, lesions_tr= self.flipping.apply_transform(feat_tr, lesions_tr) 
        #warping
        if self.warping != None:
            random_p = np.random.rand()
            self.log.debug(f'random probability for warping : {random_p}')
            if random_p < self.warping.p:
                self.log.debug('apply warping')
                feat_tr, lesions_tr= self.warping.apply_transform(feat_tr, lesions_tr)            
        return feat_tr, lesions_tr
    
   
       