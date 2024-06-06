import os
import numpy as np
import nibabel as nb
import copy
import time
from scipy import sparse
import meld_graph.mesh_tools as mt
import torch
from math import pi
import logging
from meld_graph.paths import (
    SCRIPTS_DIR,
)


class Transform:
    """Class transform paramaters TODO update doctring with a few more details"""

    def __init__(self, params_transform):
        self.p = params_transform["p"]
        self.indices = np.load(os.path.join(SCRIPTS_DIR, params_transform["file"]))
        self.indices = self.indices.astype("int")

    def get_indices(self):
        """Randomly choose a precalculated transformation"""
        transf = np.random.randint(0, len(self.indices))
        # initiate lambdas and indices to speed up
        indices = copy.deepcopy(self.indices[transf])
        return indices


class Augment:
    """Class to augment data"""

    def __init__(self, params, graph_tools):
        """Augment class TODO update docstring
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
        self.params = params
        self.transform_types = set(self.params)
        if "spinning" in self.transform_types:
            self.spinning = Transform(self.params["spinning"])
        else:
            self.spinning = None
        if "warping" in self.transform_types:
            self.warping = Transform(self.params["warping"])
        else:
            self.warping = None
        if "flipping" in self.transform_types:
            self.flipping = Transform(self.params["flipping"])
        else:
            self.flipping = None

        self.gt = graph_tools
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # need to load neighbours

    def get_p_param(self, param):
        """check pvalue, set to zero if not found"""
        if param not in list(self.params.keys()):
            return 0
        else:
            return self.params[param]["p"]

    def add_gaussian_noise(self, feat_tr):
        """add a gaussian noise"""
        variance = np.random.uniform(0, 0.1)
        feat_tr = feat_tr + np.random.normal(0.0, variance, size=feat_tr.shape)
        return feat_tr

    # TODO this does not do anything! - also remove? Alternatively document that this does not have any effect
    def add_gaussian_blur(self, feat_tr):
        """add gaussian blur function"""
        # n_iterations = np.random.choice(10)
        # for iteration in np.arange(n_iterations):
        #    feat_tr = self.smooth_step(feat_tr)
        return feat_tr

    def add_brightness_scaling(self, feat_tr):
        multipliers = np.random.uniform(0.75, 1.25, size=feat_tr.shape[1])
        feat_tr *= multipliers[None, :]
        return feat_tr

    def adjust_contrast(self, feat_tr):
        """adjust contrast"""
        for c in range(feat_tr.shape[1]):
            factor = np.random.uniform(0.65, 1.5)
            mn = feat_tr[:, c].mean()
            minm = feat_tr[:, c].min()
            maxm = feat_tr[:, c].max()
            feat_tr[:, c] = (feat_tr[:, c] - mn) * factor + mn
            feat_tr[:, c][feat_tr[:, c] < minm] = minm
            feat_tr[:, c][feat_tr[:, c] > maxm] = maxm
        return feat_tr

    # TODO delete? If yes -> delete low_res from example experiment config
    def add_low_res(self, feat_tr):
        """add low resolution version"""
        return feat_tr

    def add_gamma_scale(self, feat_tr):
        """add gamma scaling"""
        epsilon = 1e-7
        mn = feat_tr.mean(axis=0)
        sd = feat_tr.std(axis=0)
        minm = feat_tr.min(axis=0)
        rnge = feat_tr.max(axis=0) - minm
        gamma = np.random.uniform(0.7, 1.5, size=feat_tr.shape[1])  # .astype(np.float16)
        feat_tr = np.power(((feat_tr - minm) / (rnge + epsilon)), gamma) * (rnge + epsilon) + minm
        feat_tr = (feat_tr - mn) / (sd + epsilon)
        return feat_tr

    def augment_lesion(self, tdd, noise_std=0.5):
        """Modify lesion using low frequency noise."""

        # get geodesic distance (negative inside lesion, positive outside)
        # normalise by minimum values
        new_dist = tdd["distances"]
        new_dist_norm = new_dist / np.abs(new_dist.min())
        # create low frequencies noise on low res icosphere 2
        n_vert_low = len(self.gt.icospheres.icospheres[2]["coords"])
        noise = np.random.normal(0, noise_std, n_vert_low)
        # upsample noise to high res
        for level in range(2, 7):
            unpool_ind = self.gt.unpool(level=level + 1)
            noise_upsampled = unpool_ind(torch.from_numpy(noise.reshape(-1, 1)), device=None)
            # TODO Q Hannah why do we pass noise_upsampled to the CPU?
            noise_upsampled = noise_upsampled.detach().cpu().numpy().ravel()
            noise = noise_upsampled.copy()
        # add noise to distance normalised
        new_mask = (new_dist_norm + noise_upsampled) <= 0
        # print(f'no lesion before: {sum(tdd["labels"])}, no lesion after {sum(new_mask)}')
        tdd["labels"] = new_mask
        return tdd

    def recompute_distance_and_smoothed(self, tdd):
        """recompute distances from augmented lesion masks"""
        tdd["distances"] = self.gt.fast_geodesics(tdd["labels"]).astype(np.float32)
        tdd["smooth_labels"] = self.gt.smoothing(tdd["labels"], iteration=10).astype(np.float32)
        return

    def apply_indices(self, indices, tdd):
        """TODO"""
        # spin features
        for field in tdd.keys():
            # no point in spinning empty labels
            if field == "labels" or field == "smooth_labels":
                # TODO Q Hannah: potential bug here: smooth_labels could result in labels <1 everywhere. Better to do (tdd[field]==0).all()
                if (tdd[field] == 1).any():
                    tdd[field] = tdd[field][indices]
            else:
                tdd[field] = tdd[field][indices]
        return tdd

    def apply(self, subject_data_dict):
        """TODO"""
        # create a transformed data dict
        tdd = subject_data_dict.copy()
        # randomly augment lesion using distances and noise
        # NOTE lesion augmentation needs to happen before spinning, as medial wall is re-masked after new distances were calculated
        if (tdd["labels"] == 1).any():
            if np.random.rand() < self.get_p_param("augment_lesion"):
                tdd = self.augment_lesion(tdd)
                self.recompute_distance_and_smoothed(tdd)

        mesh_transform = False
        indices = np.arange(tdd["features"].shape[0], dtype=int)
        # mesh augmentations
        # stack the transformations into a single indexing step
        if np.random.rand() < self.get_p_param("spinning"):
            mesh_transform = True
            indices = indices[self.spinning.get_indices()]

        if np.random.rand() < self.get_p_param("warping"):
            mesh_transform = True
            indices = indices[self.warping.get_indices()]

        if np.random.rand() < self.get_p_param("flipping"):
            mesh_transform = True
            indices = indices[self.flipping.get_indices()]

        # apply just once
        if mesh_transform:
            tdd = self.apply_indices(indices, tdd)

        # Gaussian noise
        if np.random.rand() < self.get_p_param("noise"):
            tdd["features"] = self.add_gaussian_noise(tdd["features"])
        # Gaussian blur
        if np.random.rand() < self.get_p_param("blur"):
            tdd["features"] = self.add_gaussian_blur(tdd["features"])
        # Brightness scaling
        if np.random.rand() < self.get_p_param("brightness"):
            tdd["features"] = self.add_brightness_scaling(tdd["features"])

        # adjust contrast
        if np.random.rand() < self.get_p_param("contrast"):
            tdd["features"] = self.add_brightness_scaling(tdd["features"])
        # low res - not implemented
        if np.random.rand() < self.get_p_param("low_res"):
            tdd["features"] = self.add_low_res(tdd["features"])

        # gamma intensity
        if np.random.rand() < self.get_p_param("gamma") / 2:
            tdd["features"] = self.add_gamma_scale(tdd["features"])
        # inverted gamma intensity
        if np.random.rand() < self.get_p_param("gamma") / 2:
            tdd["features"] = -self.add_gamma_scale(-tdd["features"])

        return tdd
