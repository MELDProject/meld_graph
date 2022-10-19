#ico class
import os
import numpy as np
import nibabel as nb
import copy
from scipy import sparse 
import meld_classifier.mesh_tools as mt
import torch
from math import pi 
import logging
from meld_graph.icospheres import IcoSpheres
import matplotlib_surface_plotting as msp
from scipy.stats import special_ortho_group
from scipy.spatial import cKDTree
# import logging
### test class augment
from meld_graph.augment import Augment
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_graph.dataset import GraphDataset

icos=IcoSpheres(conv_type='SpiralConv')
ico_index=7
ico_ini = icos.icospheres[ico_index]

#test on subject
site_codes=['H4']
subjects = ['MELD_H4_3T_FCD_0011']
features=  [  '.combat.on_lh.pial.K_filtered.sm20.mgh',
        '.combat.on_lh.thickness.sm10.mgh',
        '.combat.on_lh.w-g.pct.sm10.mgh',
        '.combat.on_lh.sulc.sm5.mgh',
        '.combat.on_lh.curv.sm5.mgh',
        '.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
        '.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
        '.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
        '.combat.on_lh.gm_FLAIR_0.sm10.mgh',
        '.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
        '.combat.on_lh.wm_FLAIR_1.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh',
        '.inter_z.intra_z.combat.on_lh.thickness.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.w-g.pct.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.sulc.sm5.mgh',
        '.inter_z.intra_z.combat.on_lh.curv.sm5.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.thickness.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.sulc.sm5.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.curv.sm5.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh',]
            
# initiate params
def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config

config = load_config('../scripts/config_files/experiment_config_synth_nnunet.py')
config.data_parameters['features']=features
cohort = MeldCohort(
            hdf5_file_root=config.data_parameters["hdf5_file_root"], dataset=config.data_parameters["dataset"]
        )
dataset = GraphDataset(subjects, cohort, config.data_parameters )

params = config.data_parameters
params['augment_data'] = {'spinning': {'p': 0.0, 'file': 'data/spinning/spinning_ico7_10.npy'},
 'warping': {'p': 0.0, 'file': 'data/warping/warping_ico7_10.npy'},
 'noise': {'p': 0.15},
 'blur': {'p': 0.2},
 'brightness': {'p': 0.15},
 'contrast': {'p': 0.15},
 'low_res': {'p': 0.25},
 'gamma': {'p': 0.15},
 'flipping': {'p': 0.0, 'file': 'data/flipping/flipping_ico7_3.npy'}}
augment = Augment(params['augment_data'])

features_subj, labels_subj = dataset.data_list[0]
import time
t1=time.time()
for k in np.arange(1000):
    if k%100==0:
        t2=time.time()
        print(t2-t1)
    spinned_feature, spinned_lesion = augment.apply(features_subj, labels_subj)
print('time without spins',t2-t1)

params['augment_data'] = {'spinning': {'p': 0.2, 'file': 'data/spinning/spinning_ico7_10.npy'},
 'warping': {'p': 0.2, 'file': 'data/warping/warping_ico7_10.npy'},
 'noise': {'p': 0.15},
 'blur': {'p': 0.2},
 'brightness': {'p': 0.15},
 'contrast': {'p': 0.15},
 'low_res': {'p': 0.25},
 'gamma': {'p': 0.15},
 'flipping': {'p': 0.5, 'file': 'data/flipping/flipping_ico7_3.npy'}}
augment = Augment(params['augment_data'])

features_subj, labels_subj = dataset.data_list[0]
import time
t1=time.time()
for k in np.arange(1000):
    if k%100==0:
        t2=time.time()
        print(t2-t1)
    spinned_feature, spinned_lesion = augment.apply(features_subj, labels_subj)
print('time with spins',t2-t1)



