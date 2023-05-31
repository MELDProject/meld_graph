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


def transform_coords(icos, transform, ax=None):
    """transform coordinates: flip, warp or spin"""
    trans_coords = copy.deepcopy(icos.icospheres[7]["coords"])
    if transform == "flipping":
        trans_coords[:, ax] = -trans_coords[:, ax]
        return trans_coords
    elif transform == "spinning":
        trans_coords = spinning_coords(trans_coords)
        return trans_coords
    elif transform == "warping":
        warped_coords_2 = warp_mesh(icos.icospheres[2], warp_fraction=3)
        warped_coords_3 = upsample_mesh(warped_coords_2, icos.icospheres[2], icos.icospheres[3])
        warped_coords_4 = upsample_mesh(warped_coords_3, icos.icospheres[3], icos.icospheres[4])
        warped_coords_5 = upsample_mesh(warped_coords_4, icos.icospheres[4], icos.icospheres[5])
        warped_coords_6 = upsample_mesh(warped_coords_5, icos.icospheres[5], icos.icospheres[6])
        warped_coords_7 = upsample_mesh(warped_coords_6, icos.icospheres[6], icos.icospheres[7])
        return warped_coords_7


def find_indices(trans_coords, orig_coords):
    tree = cKDTree(trans_coords)
    distance, indices = tree.query(orig_coords, k=1)
    return indices


if __name__ == "__main__":
    # Create icospheres
    icos = IcoSpheres(conv_type="SpiralConv")
    # how many of each. flipping can only be 3
    transforms = {"flipping": 3, "warping": 10, "spinning": 10}
    for transform in transforms.keys():
        print("calculating ", transform)
        data_dir = "../data/{}".format(transform)
        file_name = f"{transform}" + "_ico7_{}.npy"
        transformed_indices = []
        n_transforms = transforms[transform]
        for n_t in range(0, n_transforms):

            # Warp icosphere at second level
            trans_coords = transform_coords(icos, transform, ax=n_t)
            # Find nearest 3 neighbours vertices from spinned ico for each vertices in initial ico
            indices = find_indices(trans_coords, icos.icospheres[7]["coords"])
            # Add to multiple arrays
            transformed_indices.append(indices)
        transformed_indices = np.array(transformed_indices)
        # Save dictionary with lambdas and indices
        output_file = os.path.join(data_dir, file_name.format(n_transforms))
        np.save(output_file, transformed_indices)
