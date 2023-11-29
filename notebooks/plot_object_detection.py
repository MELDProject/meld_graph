import meld_graph
import meld_graph.models
from meld_graph import experiment
import meld_graph.dataset
import meld_graph.data_preprocessing
import meld_graph.evaluation

import logging
import os
import json

from meld_graph.dataset import GraphDataset, Oversampler
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_graph.training import Metrics
import numpy as np
from meld_graph.paths import EXPERIMENT_PATH

from meld_graph.evaluation import Evaluator
from meld_graph.graph_tools import GraphTools
from meld_graph.icospheres import IcoSpheres
import pandas as pd
import torch_geometric


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config



fold =1
config = load_config(f'../scripts/config_files/23-06-30_TRNW_object/fold_0{fold}/s_0.py')
params = config.data_parameters


cohort= MeldCohort(hdf5_file_root=params['hdf5_file_root'], dataset=params['dataset'])

# initialise subjects manually in list or get from a csv dataset
subject_ids, trainval_ids, test_ids  = cohort.read_subject_ids_from_dataset()

subjects = ['MELD_H14_3T_FCD_0035',
 'MELD_H4_15T_FCD_0002','MELD2_H7_3T_FCD_003','MELD_H4_15T_C_0004','MELD_H4_15T_C_0005'
]
exp = experiment.Experiment(config.network_parameters, config.data_parameters)


#load dataset
dataset = GraphDataset(subjects, cohort, params, mode='test')
train_data_loader = torch_geometric.loader.DataLoader(
            dataset,
            #sampler=sampler,
            shuffle=True,
            batch_size=8,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )


exp.load_model(checkpoint_path=f'/rds/project/kw350/rds-kw350-meld/experiments_graph/kw350/23-06-30_TRNW_object/s_0/fold_0{fold}/best_model.pt',
               force=True)

for d in train_data_loader:
    break


import torch
estimates = exp.model(d.x)





#plot this
import matplotlib_surface_plotting as msp

def calculate_circle_dataset(center,radius, spherical_coords):
    """calculate circle of radius around center"""
    all_distances = np.linalg.norm(spherical_coords-center,axis=1)
    circle = all_distances<radius
    return circle,all_distances

def plot_object_detection_output(xyzr, cohort, icospheres,fname):
    sphere = icospheres.icospheres[7]
    spherical_coords = sphere['coords']/100
    mask,dists = calculate_circle_dataset(xyzr[:3],xyzr[3],
                         spherical_coords)
    msp.plot_surf(cohort.surf['coords'],
              icospheres.icospheres[7]['faces'],
              #dataset[0]['distance_map'].detach().numpy(),
              mask,
              cmap='turbo',rotate=[90,270],
              filename=fname
              )
    return

icospheres = IcoSpheres()

for i in np.arange(10):
    prediction = torch.exp(estimates['log_softmax'])[:,1].detach().numpy()
    prediction = prediction.reshape((-1,163842,))
    prediction = prediction[i]
    label=dataset[i]['y'].detach().numpy()
    if    label.sum()>0:

        msp.plot_surf(cohort.surf['coords'],
                    icospheres.icospheres[7]['faces'],
                    #dataset[0]['distance_map'].detach().numpy(),
                    prediction,
                    cmap='viridis',rotate=[90,270],
                    parcel=label,
        filename=f'preds/sub_{i}.png'
                    )

        plot_object_detection_output(estimates['object_detection_linear'].detach().numpy()[i], cohort, 
        icospheres,fname=f'preds/sub_obj_{i}.png')
