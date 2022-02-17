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
    
    def apply_spinning(self, data, labels=False):
        # select random spinning
        spin = np.random.randint(0,len(lambdas))
        data_spinned = self.lambdas[spin,:,0]*data[indices[spin,:,0]] + self.lambdas[spin,:,1]*data[indices[spin,:,1]] + self.lambdas[spin,:,2]*data[indices[spin,:,2]]
        if labels==True:
            data_spinned = np.round(data_spinned)
        return data_spinned
        
    def apply(self, data,):
        #TODO : add if condition for train/val
        random_p = np.random.rand()
        if random_p < self.p :
            if self.spinning:
                data_tr = apply_spinning(data)    
        else:
            self.log.info('no augmentation')
            data_tr = data
        return data_tr
    
   
       