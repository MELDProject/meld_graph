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


#loads in all icosphere
class Augment():
    """Class to augment data"""
    def __init__(self, params):
        """Augment class
        data - data to augment
        params - dictionary containing augmentation method, file, and probability of apply transformation (p)
        """ 
        self.log = logging.getLogger(__name__)
        if params['spinning'] != None:
            self.spinning = True
            self.p = params['spinning']['p']
            self.lambdas, self.indices = np.load(params['spinning']['file'])
            self.indices = self.indices.astype('int')
            
    def apply_spinning(self, feats, lesions=None):
        # select random spinning
        spin = np.random.randint(0,len(self.lambdas))
        # spin lesions if exist
        if lesions.any()!= None:
            lesions_spinned = self.lambdas[spin,:,0]*lesions[self.indices[spin,:,0]] + self.lambdas[spin,:,1]*lesions[self.indices[spin,:,1]] + self.lambdas[spin,:,2]*lesions[self.indices[spin,:,2]]   
            lesions_spinned = np.round(lesions_spinned)
        # spin features
        n_feat = len(feats.T)
        lambdas = np.tile(self.lambdas[:,:,:,np.newaxis], n_feat )
        feats_spinned = lambdas[spin,:,0]*feats[self.indices[spin,:,0]] + lambdas[spin,:,1]*feats[self.indices[spin,:,1]] + lambdas[spin,:,2]*feats[self.indices[spin,:,2]]        
        feats_spinned_clean=np.zeros(feats_spinned.shape)
        for i in range(0,n_feat):
            feats_spinned_clean[:,i]=np.clip(feats_spinned[:,i], np.percentile(feats_spinned[:,i], 0.01),np.percentile(feats_spinned[:,i], 99.9))  
        return feats_spinned_clean, lesions_spinned
        
    def apply(self, features, lesions=None):
        #TODO : add if condition for train/val
        random_p = np.random.rand()
        if random_p < self.p :
            if self.spinning:
                feat_tr, lesions_tr= self.apply_spinning(features, lesions)
        else:
            self.log.info('no augmentation')
            feat_tr = features
            lesions_tr = lesions
        return feat_tr, lesions_tr
    
   
       