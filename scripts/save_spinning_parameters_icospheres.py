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
from meld_graph.icospheres import IcoSpheres
import matplotlib_surface_plotting as msp
from scipy.stats import special_ortho_group
from scipy.spatial import cKDTree
import pickle


def spinning_coords(coords):
    ''' rotate coordinates '''
    rotation = special_ortho_group.rvs(3)
    new_coords = coords @ rotation
    return new_coords

def barycentric_coordinates_matrix(p,tri):
    '''solve to return coordinates as barycentric from 3 vertices of triangle.
       Use outputs for linear interpolation'''
    a = (np.square(tri[:,0,0]-tri[:,2,0]) + np.square(tri[:,0,1]-tri[:,2,1]) + np.square(tri[:,0,2]-tri[:,2,2]))
    b = (tri[:,1,0]-tri[:,2,0])*(tri[:,0,0]-tri[:,2,0]) + (tri[:,1,1]-tri[:,2,1])*(tri[:,0,1]-tri[:,2,1]) + (tri[:,1,2]-tri[:,2,2])*(tri[:,0,2]-tri[:,2,2])
    c = b
    d = (np.square(tri[:,1,0]-tri[:,2,0]) + np.square(tri[:,1,1]-tri[:,2,1]) + np.square(tri[:,1,2]-tri[:,2,2]))
    f = (p[:,0] - tri[:,2,0])*(tri[:,0,0]-tri[:,2,0]) + (p[:,1]-tri[:,2,1])*(tri[:,0,1]-tri[:,2,1]) + (p[:,2]-tri[:,2,2])*(tri[:,0,2]-tri[:,2,2])
    g = (p[:,0] - tri[:,2,0])*(tri[:,1,0]-tri[:,2,0]) + (p[:,1]-tri[:,2,1])*(tri[:,1,1]-tri[:,2,1]) + (p[:,2]-tri[:,2,2])*(tri[:,1,2]-tri[:,2,2])
    chi = (d*f - b*g)/(a*d - b*c)
    eta = (-c*f + a*g)/(a*d - b*c)
    lambda1 = chi
    lambda2 = eta
    lambda3 = 1 - chi - eta
    return np.vstack((lambda1, lambda2, lambda3)).T


if __name__ == '__main__':
    #initialise params
    ico_index=7
    num_rotation=10
    data_dir='../data/spinning'
    file_name='spinning_ico7_{}.npy'
    
    # Create icospheres
    icos=IcoSpheres(conv_type='SpiralConv')
    
    # Select icosphere
    ico_ini = icos.icospheres[ico_index]
    
    # Create multiple spinned icosphere
    spinned_lambdas = []
    spinned_indices = []
    for rot in range(0,num_rotation):
        # Spin icosphere
        ico_spinned = copy.deepcopy(ico_ini)
        ico_spinned['coords']=spinning_coords(ico_ini['coords'])
        # Find nearest 3 neighbours vertices from spinned ico for each vertices in initial ico
        tree=cKDTree(ico_spinned['coords'])
        distance, indices=tree.query(ico_ini['coords'],k=3)
        # Find lambda1, lambda2, lambda3, vectors values for barycentric
        lambdas = barycentric_coordinates_matrix(ico_ini['coords'],ico_spinned['coords'][indices])
        # Add to multiple arrays
        spinned_lambdas.append(lambdas)
        spinned_indices.append(indices)
    spinned_lambdas=np.array(spinned_lambdas)
    spinned_indices=np.array(spinned_indices)
    
    # Save dictionary with lambdas and indices
    output_file = os.path.join(data_dir,file_name.format(num_rotation))
    data=(spinned_lambdas,spinned_indices)
    np.save(output_file,data)