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
from meld_graph.resampling_meshes import *


if __name__ == "__main__":
    # initialise params
    ico_index = 7
    num_warps = 10
    data_dir = "../data/warping"
    file_name = "warping_ico7_{}.npy"

    # Create icospheres
    icos = IcoSpheres(conv_type="SpiralConv")

    # Create multiple spinned icosphere
    warped_lambdas = []
    warped_indices = []
    for rot in range(0, num_warps):
        # Warp icosphere at second level
        #         ico_index = np.random.choice(np.arange(3)+1)
        #         ico_ini = icos.icospheres[ico_index]
        warped_coords_2 = warp_mesh(icos.icospheres[2], warp_fraction=3)
        warped_coords_3 = upsample_mesh(warped_coords_2, icos.icospheres[2], icos.icospheres[3])
        warped_coords_4 = upsample_mesh(warped_coords_3, icos.icospheres[3], icos.icospheres[4])
        warped_coords_5 = upsample_mesh(warped_coords_4, icos.icospheres[4], icos.icospheres[5])
        warped_coords_6 = upsample_mesh(warped_coords_5, icos.icospheres[5], icos.icospheres[6])
        warped_coords_7 = upsample_mesh(warped_coords_6, icos.icospheres[6], icos.icospheres[7])

        # Find nearest 3 neighbours vertices from spinned ico for each vertices in initial ico
        tree = cKDTree(warped_coords_7)
        distance, indices = tree.query(icos.icospheres[7]["coords"], k=3)
        lambdas = barycentric_coordinates_matrix(icos.icospheres[7]["coords"], warped_coords_7[indices])
        redos = np.where(~np.logical_and(0 < lambdas, lambdas < 1).all(axis=1))[0]
        # fix the ones that aren't quite right
        indices, lambdas = correct_triangles(icos.icospheres[7], indices, redos, warped_coords_7, lambdas)
        # Add to multiple arrays
        warped_lambdas.append(lambdas)
        warped_indices.append(indices)
    warped_lambdas = np.array(warped_lambdas)
    warped_indices = np.array(warped_indices)

    # Save dictionary with lambdas and indices
    output_file = os.path.join(data_dir, file_name.format(num_warps))
    data = (warped_lambdas, warped_indices)
    np.save(output_file, data)
