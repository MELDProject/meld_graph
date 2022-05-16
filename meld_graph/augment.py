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
        for i in range(0,n_feat):
            feats_transf_clean[:,i]=np.clip(feats_transf[:,i], np.percentile(feats_transf[:,i], 0.01),np.percentile(feats_transf[:,i], 99.9))  
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
        if 'warp' in self.transform_types:
            self.warp = Transform(self.params['warp'])
        else:
            self.warp = None
       
    def apply(self, features, lesions=None):
        #spin
        if self.spinning != None:
            random_p = np.random.rand()
            self.log.debug(f'random probability : {random_p}')
            if random_p < self.spinning.p:
                self.log.debug('apply spinning')
                feat_tr, lesions_tr= self.spinning.apply_transform(features, lesions)
        #warp
        if self.warp != None:
            random_p = np.random.rand()
            self.log.debug(f'random probability : {random_p}')
            if random_p < self.warp.p:
                self.log.debug('apply warp')
                feat_tr, lesions_tr= self.warp.apply_transform(features, lesions) 
        else:
            self.log.info('no augmentation')
            feat_tr = features
            lesions_tr = lesions
        return feat_tr, lesions_tr
    
   
       