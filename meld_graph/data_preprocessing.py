from meld_graph.paths import (
    BASE_PATH,
    NVERT,
    DEMOGRAPHIC_FEATURES_FILE,
    DK_ATLAS_FILE,
    MELD_PARAMS_PATH,
)
import numpy as np
import nibabel as nb
import os
import logging
import json
import csv
import copy
import h5py
import pandas as pd
import sys
import pickle
import random
from itertools import chain
from meld_graph.meld_cohort import MeldSubject, MeldCohort
from neuroCombat import neuroCombat, neuroCombatFromTraining
import meld_graph.distributedCombat as dc
import meld_graph.mesh_tools as mt
import meld_graph.meld_plotting as mpt

class Preprocess:
    """
    Load and preprocess data.

    params:
        scaling: scale data between 0 and 1 using precomputed scaling params (TODO currently not implemented)
        zscore: z-score each feature for each subject (excluding medial wall)

    TODO : transform data if not gaussian
    TODO : if not using combat features, correct for sulc freesurfer
    """

    params = {"scaling": None, "zscore": False}

    def __init__(
        self,
        cohort,
        site_codes=None,
        write_output_file=None,
        icospheres=None,
        data_dir=BASE_PATH,
        meld_dir=MELD_PARAMS_PATH,
        params={},
    ):
        self.cohort = cohort
        self.write_output_file = write_output_file
        self.data_dir = data_dir
        self.params.update(params)
        # private attributes for site_codes and subject_ids properties
        self._site_codes = site_codes
        self._subject_ids = None
        self.log = logging.getLogger(__name__)
        self._lobes = None
        self.initialise_distances()
        self.icospheres = icospheres
        if self.params["zscore"] != False:
            self.load_z_params(os.path.join(MELD_PARAMS_PATH,self.params["zscore"]))
        self.feat = Feature()
        # calibration_smoothing : curve to calibrate smoothing on surface mesh
        self._calibration_smoothing = None
        self.meld_dir = meld_dir
    @property
    def site_codes(self):
        if self._site_codes is None:
            self._site_codes = self.cohort.get_sites()
        return self._site_codes

    @property
    def subject_ids(self):
        if self._subject_ids is None:
            # filter subject ids based on site codes
            self._subject_ids = self.cohort.get_subject_ids(site_codes=self.site_codes, lesional_only=False)
        return self._subject_ids

    @property
    def lobes(self):
        if self._lobes is None:
            self._lobes = self.load_lobar_parcellation()
        return self._lobes

    def load_lobar_parcellation(self, lobe=1):
        parc = nb.freesurfer.io.read_annot(os.path.join(self.data_dir, "fsaverage_sym", "label", "lh.lobes.annot"))[0]
        lobes = (parc == lobe).astype(int)
        return lobes

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def save_cohort_features(self, feature_name, features, subject_ids, hemis=["lh", "rh"]):
        assert len(features) == len(subject_ids)
        for s, subject in enumerate(subject_ids):
            subj = MeldSubject(subject, cohort=self.cohort)
            subj.write_feature_values(
                feature_name,
                features[s],
                hemis=hemis,
                hdf5_file_root=self.write_output_file,
            )

    # def correct_sulc_freesurfer(self, vals):
    #     """this function normalized sulcul feature in cm when values are in mm (depending on Freesurfer version used)"""
    #     if np.mean(vals, axis=0) > 0.2:
    #         vals = vals / 10
    #     else:
    #         pass
    #     return vals
    
    def correct_sulc_freesurfer(self, vals, mask):
        """this function normalized sulcul feature in cm when values are in mm (depending on Freesurfer version used)"""
        if np.mean(abs(vals)[mask], axis=0) > 2:
            vals = vals / 10
        else:
            pass
        return vals

    def plot_subject_features(self, features_to_plot):
        """plot subject features in a given output folder"""
        for subj_id in self.subject_ids:
            subj = MeldSubject(subj_id, self.cohort)
            
            if not subj.has_flair:
                features_to_plot_subj = [feature for feature in features_to_plot if not 'FLAIR' in feature]
            else:
                features_to_plot_subj = features_to_plot
            # create output folder if does not exist
            os.makedirs(os.path.join(BASE_PATH, f'MELD_{subj.site_code}', "images"), exist_ok=True)
            
            hemis = ["lh", "rh"]
            for hemi in hemis:
                features_values = []
                for feature in features_to_plot_subj:
                    feature_values = subj.load_feature_values(feature, hemi=hemi)
                    features_values.append(feature_values)
                mpt.plot_single_subject(
                    features_values,
                    lesion=None,
                    feature_names=features_to_plot_subj,
                    out_filename=os.path.join(BASE_PATH, f'MELD_{subj.site_code}', "images", f"qc_features_{subj_id}_{hemi}.jpeg"),
                )
            
    def get_data_preprocessed(
        self,
        subject,
        features,
        lobes=False,
        lesion_bias=False,
        distance_maps=False,
        combine_hemis=None,
    ):
        """
        Preprocess features data for a single subject depending on params.

        Args:
            lobes: if True, return lobes task as lesion values
            lesion_bias: if True, add lesion_bias value to lesional vertices.
                NOTE: should not be used for final models, only for testing
                NOTE: this is an old flag, not in use in the current model anymore.
            distance_maps: read precalculated distance maps from subject.
                NOTE: this is an old flag, not in use anymore.
                Distance are now calulated on the fly (because of lesion augmentation).
            combine_hemis: combine hemispheres to one sample by stacking.

        Returns:
            features_left, features_right, lesion_left, lesion_right
        """
        subj = MeldSubject(subject, cohort=self.cohort)
        subject_data = []
        # load data & lesion
        for hemi in ("lh", "rh"):
            vals_array, lesion = subj.load_feature_lesion_data(features, hemi=hemi)
            subject_data_dict = {}
            # z-score data
            if self.params["zscore"]:
                if hemi == "lh":
                    self.log.info(f"Z-scoring data for {subject}")
                vals_array = self.zscore_data(vals_array.T, features).T
            if distance_maps:
                gdist = self.load_distances(subj, hemi)
                subject_data_dict["distances"] = gdist
            if self.params["scaling"] is not None:
                self.log.info(f"Scaling data for has been removed. REIMPLEMENT")
            if lobes:
                # replace lesion data with lobes task if required
                lesion = self.lobes
            # add lesion bias
            if lesion_bias:
                self.log.info(f"WARNING: Adding lesion bias of {lesion_bias} to {subject}")
                vals_array[lesion == 1] += lesion_bias
            if combine_hemis is not None:
                self.log.info(f"WARNING: Combine_hemis is not implemented.")

            subject_data_dict["features"] = vals_array
            subject_data_dict["labels"] = lesion

            subject_data.append(subject_data_dict)
        return subject_data

    def load_distances(self, subj, hemi="lh"):
        """load geodesic distance from lesion or 300s"""
        if (not subj.is_patient) or (subj.get_lesion_hemisphere() != hemi):
            gdist = np.ones(NVERT, dtype=np.float32) * 300

        else:
            gdist = subj.load_feature_values(".on_lh.boundary_zone.mgh", hemi=hemi)
            # threshold to range 0,300
            gdist = np.clip(gdist, 0, 300)
        return gdist

    def load_z_params(self, file="data/feature_means.json"):
        import json

        with open(file, "r") as fp:
            self.z_params = json.load(fp)

    def zscore_data(self, values, features):
        """zscore features using precalculated means and stds"""
        for fi, f_value in enumerate(values):
            if np.std(f_value) != 0:
                values[fi] = (f_value - self.z_params[features[fi]]["mean"]) / self.z_params[features[fi]]["std"]
        return values

    def scale_data(self, matrix, features, file_name):
        """scale data features between 0 and 1"""
        self.log.info(f"Scale data using file {file_name}")
        file = os.path.join(BASE_PATH, file_name)
        with open(file, "r") as f:
            params_norm = json.loads(f.read())
        data = copy.deepcopy(matrix)
        # scale and clip data
        for f, feature in enumerate(features):
            min_val = float(params_norm[feature]["min"])
            max_val = float(params_norm[feature]["max"])
            data[f] = np.clip(data[f], min_val, max_val)
            data[f] = (data[f] - min_val) / (max_val - min_val)
        return data

    def compute_scaling_parameters(self, feature):
        """get mean and std of all brain for the given cohort and save parameters"""
        print(f"Compute scaling values for feature {feature}")
        # Give warning if list of subjects empty
        if len(self.subject_ids) == 0:
            print("WARNING: There is no subject in this cohort")
        vals_array = []
        included_subj = []
        for id_sub in self.subject_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(
                    np.hstack(
                        [
                            vals_lh[self.cohort.cortex_mask],
                            vals_rh[self.cohort.cortex_mask],
                        ]
                    )
                )
                if feature == ".on_lh.sulc.mgh":
                    vals = self.correct_sulc_freesurfer(vals)
                vals_array.append(vals)
                included_subj.append(id_sub)
            else:
                pass
        self.log.info("INFO: Compute min and max from {} subjects".format(len(included_subj), feature))
        # get min and max percentile
        vals_array = np.array(vals_array)
        min_val = np.percentile(vals_array.flatten(), 1, axis=0)
        max_val = np.percentile(vals_array.flatten(), 99, axis=0)
        self.log.info(f"min value = {min_val}")
        self.log.info(f"max value = {max_val}")
        # save in json
        data = {}
        data["{}".format(feature)] = {
            "min": str(min_val),
            "max": str(max_val),
        }
        # create or re-write json file
        file = os.path.join(self.data_dir, self.write_output_file)
        if os.path.isfile(file):
            # open json file and get dictionary
            with open(file, "r") as f:
                x = json.loads(f.read())
            # update dictionary with new dataset version
            x.update(data)
        else:
            x = data
        # save dictionary in json file
        with open(file, "w") as outfile:
            json.dump(x, outfile, indent=4)
        print(f"INFO: Parameters saved in {file}")

    def clockwiseangle_and_distance(self, point, origin):
        import math

        refvec = [0, 1]
        # Vector between point and the origin: v = p - o
        vector = [point[0] - origin[0], point[1] - origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    def generate_synthetic_data(
        self,
        coords,
        n_features,
        synth_params,
        histo_type_seed=0,
        features=None,
    ):
        """coords - spherical coordinates
        n_features - number of input features
        bias  - mean of the difference in biases
        radius - mean size of lesion
        histo_type_seed - randomly generate different histologies.
        proportion_features_abnormal - proportion of features abnormal
        smooth_lesion - smooth edge of lesions"""
        # create a histological signature of -1,0,1 on which features are abnormal
        n_verts = len(coords)
        lesion = np.zeros(n_verts, dtype=int)
        if features is None:
            features = np.random.normal(0, 1, (n_verts, n_features))

        if np.random.random() < synth_params["proportion_hemispheres_lesional"]:
            synth_dict = self.add_lesion(
                features,
                coords,
                n_features,
                synth_params,
                histo_type_seed,
            )
        else:
            synth_dict = {
                "features": features,
                "labels": lesion,
            }

        return synth_dict

    def clip_spherical_coords(self, coordinates):
        """make sure spherical coords in range"""
        coordinates[:, 0] = np.clip(coordinates[:, 0], -np.pi / 2, np.pi / 2)
        coordinates[:, 1] = np.clip(coordinates[:, 1], -np.pi, np.pi)
        return coordinates

    def initialise_distances(self, res=1000):
        """function to precalculate pairwise differences
        res - resolution, higher resolutions lead to slower synthetic generation
        lower resolutions give pixelated lesions"""
        from sklearn.metrics import pairwise_distances

        self.xnew = np.linspace(-np.pi / 2, np.pi / 2, res)
        self.ynew = np.linspace(-np.pi, np.pi, res * 2)
        # calculate the shape of the grid
        self.gridshape = (2 * res, res)
        # meshgrid, weird indexing required for the interpolator function
        self.grid_coords_grid = np.meshgrid(self.xnew, self.ynew, indexing="ij")
        self.grid_coords = np.vstack([self.grid_coords_grid[0].ravel(), self.grid_coords_grid[1].ravel()]).T

        self.origin = np.array([0, 0])
        self.distances = pairwise_distances(self.origin.reshape(-1, 1).T, self.grid_coords, metric="haversine")[0]
        return

    def create_lesion_mask(self, radius, cartesian_coords, return_smoothed=True):
        """create irregular polygon lesion mask"""
        import matplotlib.path as mpltPath

        # from sklearn.metrics import pairwise_distances
        from scipy import interpolate, ndimage
        import copy
        from meld_graph.resampling_meshes import spinning_coords
        from meld_graph import mesh_tools as mt

        spun_coords = spinning_coords(cartesian_coords)
        spherical_coords = mt.spherical_np(spun_coords)[:, 1:]
        spherical_coords[:, 0] = spherical_coords[:, 0] - np.pi / 2
        spherical_coords = self.clip_spherical_coords(spherical_coords)

        # select a radius
        f_radius = np.clip(np.random.normal(radius, radius / 2), 0.05, 2)
        n_points = np.random.choice(6) + 4
        subset = self.grid_coords[self.distances < f_radius]
        # establish mask and mask coordinates
        x_mask = np.logical_and(self.grid_coords_grid[0] > -f_radius, self.grid_coords_grid[0] < f_radius)
        y_mask = np.logical_and(self.grid_coords_grid[1] > -f_radius, self.grid_coords_grid[1] < f_radius)
        grid_mask = np.logical_and(x_mask, y_mask)
        mask_shape = (x_mask.any(axis=1).sum(), y_mask.any(axis=0).sum())
        masked_grid_coords = np.vstack([self.grid_coords_grid[0][grid_mask], self.grid_coords_grid[1][grid_mask]]).T
        # make sure there are enough lesional vertices
        lesional_verts = -1
        while lesional_verts < 1:
            poly_i = np.random.choice(len(subset), n_points)
            polygon = subset[poly_i]
            polygon = np.array(
                sorted(
                    polygon,
                    key=lambda point: self.clockwiseangle_and_distance(point, self.origin),
                )
            )
            path = mpltPath.Path(polygon)
            lesion = path.contains_points(masked_grid_coords)
            # lesion = path.contains_points(self.grid_coords)
            arr_lesion = lesion.reshape(mask_shape).astype(float)
            # arr_lesion = lesion.reshape(self.gridshape,order='f').astype(float)
            # interpolate to coordinates

            full_lesion = np.zeros(self.gridshape, dtype=float)
            full_lesion[grid_mask.T] = arr_lesion.T.ravel()
            f_near = interpolate.RegularGridInterpolator((self.xnew, self.ynew), full_lesion.T, method="nearest")
            interpolated_lesion = f_near(spherical_coords)
            lesional_verts = interpolated_lesion.sum()
        # smoothed mask
        if return_smoothed:
            smoothed = ndimage.gaussian_filter(arr_lesion, 10)
            full_lesion[grid_mask.T] = smoothed.T.ravel()

            f_lin = interpolate.RegularGridInterpolator((self.xnew, self.ynew), full_lesion.T, method="linear")
            # return grid_coords,smoothed
            interpolated_smoothed = f_lin(spherical_coords)
            return interpolated_lesion, interpolated_smoothed
        else:
            return interpolated_lesion

    def sigmoid_dists(self, dists):
        m = dists == 0
        z = 1 / (1 + np.exp(-dists * 5))
        z[m] = 0
        return z

    def create_fingerprint(self, n_features, histo_type_seed, proportion_features_abnormal):
        """creates a vector of biases, seeded by the histological subtype integer.
        this is multiplied by the controlled bias term"""
        rng = np.random.default_rng(histo_type_seed)
        feature_mask = rng.random(n_features) < proportion_features_abnormal
        histo_bias_multipliers = rng.random(size=n_features)
        histo_signature = rng.integers(low=0, high=2, size=n_features) * 2 - 1
        fingerprint = feature_mask * histo_bias_multipliers * histo_signature
        return fingerprint

    def sample_fingerprint(self, fingerprint, jitter_factor):
        """use fingerprint as starting point for generating a slightly jittered individual fingerprint"""
        sampled_fingerprint = np.zeros_like(fingerprint)
        for fi, f in enumerate(fingerprint):
            if f != 0:
                sampled_fingerprint[fi] = np.random.normal(f, np.abs(f) / jitter_factor)
        return sampled_fingerprint

    def add_lesion(
        self,
        features,
        coords,
        n_features,
        synth_params,
        histo_type_seed,
    ):
        """superimpose a synthetic lesion on input data"""
        # create lesion mask
        if synth_params["smooth_lesion"]:
            lesion, smoothed_lesion = self.create_lesion_mask(synth_params["radius"], coords, return_smoothed=True)
        else:
            lesion = smoothed_lesion = self.create_lesion_mask(synth_params["radius"], coords, return_smoothed=False)
        lesion[~self.cohort.cortex_mask] = 0
        smoothed_lesion[~self.cohort.cortex_mask] = 0
        # bias is sampled from a normal dist so that some subjects are easier than others.
        sampled_bias = np.clip(
            np.random.normal(
                synth_params["bias"],
                synth_params["bias"] / synth_params["jitter_factor"],
            ),
            0,
            100,
        )
        # histo_signature - controls which features, how important and what sign
        fingerprint = self.create_fingerprint(n_features, histo_type_seed, synth_params["proportion_features_abnormal"])

        sampled_fingerprint = self.sample_fingerprint(fingerprint, synth_params["jitter_factor"])
        lesion_tiled = np.tile(smoothed_lesion.reshape(-1, 1), n_features)
        synth_bias_features = lesion_tiled * sampled_fingerprint * sampled_bias
        # apply synthetic lesion only on non-null feature
        apply = np.array([1 if feature.any() != 0 else 0 for feature in features.T])
        features = features + synth_bias_features * apply
        synth_dict = {
            "features": features.astype("float32"),
            "labels": lesion.astype("int32"),
        }

        return synth_dict

    @property
    def covars(self):
        if self._covars is None:
            self._covars = self.load_covars()
        return self._covars
    
    def transfer_lesion(self):
        new_cohort = MeldCohort(hdf5_file_root=self.write_output_file)
        new_listids = new_cohort.get_subject_ids(lesional_only=False)
        for subject in self.subject_ids:
            if subject in new_listids:
                print("exist")
                subj = MeldSubject(subject, self.cohort)
                hemi = subj.get_lesion_hemisphere()
                if hemi is not None:
                    print(f"INFO - {subj.subject_id}: Transfer lesion")
                    lesion = subj.load_feature_values(".on_lh.lesion.mgh", hemi)
                    subj.write_feature_values(
                        ".on_lh.lesion.mgh",
                        lesion[self.cohort.cortex_mask],
                        hemis=[hemi],
                        hdf5_file_root=self.write_output_file,
                    )
        
    def make_boundary_zones(self, smoothing=0, boundary_feature_name=".on_lh.boundary_zone.mgh"):
        import potpourri3d as pp3d
        # preload geodesic distance solver
        solver = pp3d.MeshHeatMethodDistanceSolver(self.cohort.surf["coords"], self.cohort.surf["faces"])

        for ids in self.subject_ids:
            subj = MeldSubject(ids, cohort=self.cohort)
            hemi = subj.get_lesion_hemisphere()
            if hemi:
                if subj.has_lesion:
                    print(ids)
                    overlay = np.round(subj.load_feature_values(hemi=hemi, feature=".on_lh.lesion.mgh")[:])
                    # smooth a bit for registration, conservative masks etc.
                    if smoothing > 0:
                        overlay = np.ceil(mt.smoothing_fs(overlay, fwhm=smoothing)).astype(int)
                    non_lesion_and_neighbours = self.flatten(np.array(self.cohort.neighbours)[overlay == 0])
                    lesion_boundary_vertices = np.setdiff1d(non_lesion_and_neighbours, np.where(overlay == 0)[0])
                    boundary_distance = solver.compute_distance_multisource(lesion_boundary_vertices)
                    # include lesion
                    boundary_distance[overlay == 1] = 0
                    # mask medial wall
                    boundary_distance = boundary_distance[self.cohort.cortex_mask]
                    # write in hdf5
                    subj.write_feature_values(
                        boundary_feature_name, boundary_distance, hemis=[hemi], hdf5_file_root=self.write_output_file
                    )
                else:
                    print("skipping ", ids)
                    
    def load_covars(self, subject_ids=None, demographic_file=DEMOGRAPHIC_FEATURES_FILE):
        # if not os.path.isfile(demographic_file):
        if subject_ids is None:
            subject_ids = self.subject_ids
        covars = pd.DataFrame()
        ages = []
        sex = []
        group = []
        sites_scanners = []
        for subject in subject_ids:
            subj = MeldSubject(subject, cohort=self.cohort)
            a, s = subj.get_demographic_features(["Age at preop", "Sex"], csv_file = demographic_file)
            ages.append(a)
            if s=='male':
                sex.append(1)
            elif s == 'female':
                sex.append(0)
            elif (s==0) or (s==1):
                sex.append(s)
            else:
                print(f'ERROR: There is an issue with the coded sex of subject {subject}')
            group.append(subj.is_patient)
            sites_scanners.append(subj.site_code) # just site code now
            
        covars["ages"] = ages
        covars["sex"] = sex
        covars["group"] = group
        covars["site_scanner"] = sites_scanners
        covars["ID"] = subject_ids

        #clean missing values in demographics
        covars["ages"] = covars.groupby("site_scanner").transform(lambda x: x.fillna(x.mean()))["ages"]
        covars["sex"] = covars.groupby("site_scanner").transform(lambda x: x.fillna(random.choice([0, 1])))["sex"]
        return covars

    def save_norm_combat_parameters(self, feature, estimates, hdf5_file):
        """Save estimates from combat and normalisation parameters on hdf5"""
        if not os.path.isfile(hdf5_file):
            hdf5_file_context = h5py.File(hdf5_file, "a")
        else:
            hdf5_file_context = h5py.File(hdf5_file, "r+")

        with hdf5_file_context as f:
            list_params = list(set(estimates))
            for parameter_name in list_params:
                parameter = estimates[parameter_name]
                parameter = np.array(parameter)
                dtype = parameter.dtype
                dtype = parameter.dtype

                group = f.require_group(feature)
                if dtype == "O":
                    dset = group.require_dataset(
                        parameter_name, shape=np.shape(parameter), dtype="S10", compression="gzip", compression_opts=9
                    )
                    dset.attrs["values"] = list(parameter)
                else:
                    dset = group.require_dataset(
                        parameter_name, shape=np.shape(parameter), dtype=dtype, compression="gzip", compression_opts=9
                    )
                    dset[:] = parameter
    
    def read_norm_combat_parameters(self, feature, hdf5_file):
        """reconstruct estimates dictionnary from the combat parameters hdf5 file"""
        hdf5_file_context = h5py.File(hdf5_file, "r")
        estimates = {}
        with hdf5_file_context as f:
            feat_dir = f[feature]
            parameters = feat_dir.keys()
            for param in parameters:
                if feat_dir[param].dtype == "S10":
                    estimates[param] = feat_dir[param].attrs["values"].astype(np.str)
                else:
                    estimates[param] = feat_dir[param][:]
        return estimates
    
    def shrink_combat_estimates(self, estimates):
        """ shrink combat estimates to reduce size file"""
        #combined mod.mean with stand.mean
        stand_mean =  estimates['stand.mean'][:, 0] + estimates['mod.mean'].mean(axis=1)
        estimates['stand.mean'] = stand_mean
        #save the number of subjects to un-shrink later
        estimates['num_subjects']= np.array([estimates['mod.mean'].shape[1]])
        #remove mod.mean to reduce estimates size
        del estimates['mod.mean']
        return estimates

    def unshrink_combat_estimates(self, estimates):
        """ unshrink combat estimates to use as input in neuroCombatFromTraining"""
        num_subjects = estimates['num_subjects'][0]
        mod_mean = np.zeros((len(estimates['stand.mean']),num_subjects))
        estimates['mod.mean'] = mod_mean
        estimates['stand.mean'] = np.tile(estimates['stand.mean'], (num_subjects,1)).T
        return estimates
    
    def combat_whole_cohort(self, feature_name, outliers_file=None, combat_params_file=None):
        """Harmonise data between site/scanner with age, sex and disease status as covariate
        using neuroComBat (Fortin et al., 2018, Neuroimage) and save in hdf5
        Args:
            feature_name (str): name of the feature, usually smoothed data.
            outliers_file (str): file name of the csv containing subject ID to exclude from harmonisation

        Returns:
            estimates : Combat parameters used for the harmonisation. Need to save for new patient harmonisation.
            info : dictionary of information from combat
        """
        # read morphological outliers from cohort.
        if outliers_file is not None:
            outliers = list(pd.read_csv(os.path.join(self.data_dir, outliers_file), header=0)["ID"])
        else:
            outliers = []
        # load in features using cohort + subject class
        combat_subject_include = np.zeros(len(self.subject_ids), dtype=bool)
        precombat_features = []
        for k, subject in enumerate(self.subject_ids):
            subj = MeldSubject(subject, cohort=self.cohort)
            # exclude outliers and subject without feature
            if (subj.has_features(feature_name)) & (subject not in outliers):
                lh = subj.load_feature_values(feature_name, hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                combat_subject_include[k] = True
            else:
                print("INFO - {subj.subject_id}: Excluded because no feature")
                combat_subject_include[k] = False
        if precombat_features:
            precombat_features = np.array(precombat_features)
            # load in covariates - age, sex, group, site and scanner unless provided
            covars = self.covars[combat_subject_include].copy()
            # check for nan
            index_nan = pd.isnull(covars).any(1).to_numpy().nonzero()[0]
            if len(index_nan) != 0:
                print(
                    "There is missing information in the covariates for subjects {}. \
                Combat aborted".format(
                        np.array(covars["ID"])[index_nan]
                    )
                )
            else:
                # function to check for single subjects
                covars, precombat_features = self.remove_isolated_subs(covars, precombat_features)
                covars = covars.reset_index(drop=True)

                dict_combat = neuroCombat(
                    precombat_features.T,
                    covars,
                    batch_col="site_scanner",
                    categorical_cols=["sex", "group"],
                    continuous_cols="ages",
                )
                # save combat parameters
                if combat_params_file is not None:
                    shrink_estimates = self.shrink_combat_estimates(dict_combat["estimates"])
                    self.save_norm_combat_parameters(feature_name, shrink_estimates, combat_params_file)

                post_combat_feature_name = self.feat.combat_feat(feature_name)

                print("INFO: Combat finished. Saving data")
                self.save_cohort_features(post_combat_feature_name, dict_combat["data"].T, np.array(covars["ID"]))
        else:
            print('INFO: No data to combat harmonised')
            pass
    
    def get_combat_new_site_parameters(
        self,
        feature,
        demographic_file,
    ):
        """Harmonise new site data to post-combat whole cohort and save combat parameters in
        new hdf5 file. 
        Args:
            feature_name (str): name of the feature

        """
        site_code=self.site_codes[0]
        site_combat_path = os.path.join(self.data_dir,f'MELD_{site_code}','distributed_combat')
        if not os.path.isdir(site_combat_path):
            os.makedirs(site_combat_path)
        meld_combat_path = os.path.join(self.meld_dir,'distributed_combat')
        listids = self.subject_ids    
        site_codes = np.zeros(len(listids))
        precombat_features=[]
        combat_subject_include = np.zeros(len(listids), dtype=bool)
        demos=[]
        for k, subject in enumerate(listids):
            # get the reference index and cohort object for the site, 0 whole cohort, 1 new cohort
            site_code_index = site_codes[k]
            subj = MeldSubject(subject, cohort=self.cohort)
            # exclude outliers and subject without feature
            if (subj.has_features(feature)) :
                lh = subj.load_feature_values(feature, hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature, hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                combat_subject_include[k] = True
            else:
                combat_subject_include[k] = False 
        if len(np.array(listids)[np.array(combat_subject_include)])==0:
            print(f'WARNING: Cannot compute harmonisation for {feature} because no subject found with this feature')
            return
        # load in covariates - age, sex, group, site and scanner unless provided    
        new_site_covars = self.load_covars(subject_ids=np.array(listids)[np.array(combat_subject_include)], demographic_file=demographic_file).copy()
        # check site_scanner codes are the same for all subjects
        if len(new_site_covars['site_scanner'].unique())==1:
            site_scanner = new_site_covars['site_scanner'].unique()[0]
        else:
            print('ERROR: Subjects on the list come from different site or scanner.\
            Make sure all your subject come from same site and scanner for the harmonisation process')
            sys.exit()
        bat = pd.Series(pd.Categorical(np.array(new_site_covars['site_scanner']),
                    categories=['H0', site_scanner]))       
        # apply distributed combat
        print('step1')
        new_site_data = np.array(precombat_features).T 
        dc.distributedCombat_site(new_site_data,
                                  bat, 
                                  new_site_covars[['ages','sex','group']], 
                                  file=os.path.join(site_combat_path,f"{site_code}_{feature}_summary.pickle"), 
                              ref_batch = 'H0', 
                              robust=True,)
        print('step2')
        dc_out = dc.distributedCombat_central(
            [os.path.join(meld_combat_path,f'MELD_{feature}.pickle'),
             os.path.join(site_combat_path,f"{site_code}_{feature}_summary.pickle")], ref_batch = 'H0'
        )
        # third, use variance estimates from full MELD cohort
        dc_out['var_pooled'] = pd.read_pickle(os.path.join(meld_combat_path,f'MELD_{feature}_var.pickle')).ravel()
        for c in ['ages','sex','group']:
            new_site_covars[c]=new_site_covars[c].astype(np.float64)      
        print('step3')
        pickle_file = os.path.join(site_combat_path,f"{site_code}_{feature}_harmonisation_params_test.pickle")
        _=dc.distributedCombat_site(
            pd.DataFrame(new_site_data), bat, new_site_covars[['ages','sex','group']], 
            file=pickle_file,
             central_out=dc_out, 
            ref_batch = 'H0', 
            robust=True,
        )
        #open pickle, shrink estimates and save in hdf5 and delete pickle
        with open(pickle_file, 'rb') as f:
            params = pickle.load(f)
        #filter name keys
        target_dict = {'batch':'batches', 'delta_star':'delta.star', 'var_pooled':'var.pooled',
           'gamma_star':'gamma.star', 'stand_mean':'stand.mean', 'mod_mean': 'mod.mean', 
           'parametric': 'del', 'eb':'del', 'mean_only':'del', 'mod':'del', 'ref_batch':'del', 'beta_hat':'del', 
          }
        estimates = params['estimates'].copy()
        for key in target_dict.keys():  
            if target_dict[key]=='del':
                estimates.pop(key)
            else:
                estimates[target_dict[key]] = estimates.pop(key)
        for key in estimates.keys():
            if key in ['a_prior', 'b_prior', 't2', 'gamma_bar']:
                estimates[key]=[estimates[key]]
            if key == 'batches':
                estimates[key]=np.array([estimates[key][0]]).astype('object')
            if key=='var.pooled':
                estimates[key]=estimates[key][:,np.newaxis]
            if key in ['gamma.star', 'delta.star']:
                estimates[key]=estimates[key][np.newaxis,:]
            estimates[key] = np.array(estimates[key])      
        #shrink estimates
        shrink_estimates = self.shrink_combat_estimates(estimates)
        #save estimates and delete pickle file
        combat_params_file=os.path.join(self.data_dir, self.write_output_file.format(site_code=site_code))
        self.save_norm_combat_parameters(feature, shrink_estimates, combat_params_file)
        os.remove(pickle_file)
        pickle_file = os.path.join(site_combat_path,f"{site_code}_{feature}_summary.pickle")
        os.remove(pickle_file)
        return estimates, shrink_estimates

    def combat_new_subject(self, feature_name, combat_params_file):
        """Harmonise new subject data with Combat parameters from whole cohort
            and save in new hdf5 file
        Args:
            subjects (list of str): list of subjects ID to harmonise
            feature_name (str): name of the feature, usually smoothed data.
            combat_estimates (arrays): combat parameters used for the harmonisation
        """
        # load combat parameters        
        precombat_features = []
        site_scanner = []
        subjects_included=[]
        for subject in self.subject_ids:
            subj = MeldSubject(subject, cohort=self.cohort)
            if subj.has_features(feature_name):
                lh = subj.load_feature_values(feature_name, hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                site_scanner.append(subj.site_code) # just site code now
                subjects_included.append(subject)
        #if matrix empty, pass
        if precombat_features:
            combat_estimates = self.read_norm_combat_parameters(feature_name, combat_params_file)
            combat_estimates = self.unshrink_combat_estimates(combat_estimates)
            combat_estimates["batches"] = [x.split('_')[0] for x in combat_estimates["batches"]] # remove scanner strenght from the batch code if exist
            precombat_features = np.array(precombat_features)
            site_scanner = np.array(site_scanner)
            dict_combat = neuroCombatFromTraining(dat=precombat_features.T, batch=site_scanner, estimates=combat_estimates)
            #check no empty or nan data after combat
            check_null = (dict_combat["data"].T==0).all(axis=1)
            check_nan = (np.isnan(dict_combat["data"].T)).all(axis=1)
            if (check_null).any() or (check_nan).any():
                subjects_error = np.array(subjects_included)[check_null | check_nan]
                print(f'ERROR: There was an error in the harmonisation of {subjects_error}')
                sys.exit()
            post_combat_feature_name = self.feat.combat_feat(feature_name)
            print("INFO: Combat finished. Saving data")
            self.save_cohort_features(post_combat_feature_name, dict_combat["data"].T, np.array(subjects_included))
        else:
            print("INFO: No data to combat harmonised")
            pass
    
    def transfer_features_no_combat(self, feature_name):
        # load combat parameters        
        precombat_features = []
        subjects_included=[]
        for subject in self.subject_ids:
            subj = MeldSubject(subject, cohort=self.cohort)
            if subj.has_features(feature_name):
                lh = subj.load_feature_values(feature_name, hemi="lh")[self.cohort.cortex_mask]
                rh = subj.load_feature_values(feature_name, hemi="rh")[self.cohort.cortex_mask]
                combined_hemis = np.hstack([lh, rh])
                precombat_features.append(combined_hemis)
                subjects_included.append(subject)
        #if matrix empty, pass
        if precombat_features:
            precombat_features = np.array(precombat_features)
            post_combat_feature_name = self.feat.combat_feat(feature_name)
            print("INFO: Transfer finished \n Saving data")
            self.save_cohort_features(post_combat_feature_name, precombat_features, np.array(subjects_included))
        else:
            print('INFO: No data to transfer')
            pass
                    
    def remove_isolated_subs(self, covars, precombat_features):
        """remove subjects where they are sole examples from the site (for FLAIR)"""

        df = pd.DataFrame(covars.groupby("site_scanner").count()["ages"])
        single_subject_sites = list(df.index[covars.groupby("site_scanner").count()["ages"] == 1])
        mask = np.zeros(len(covars)).astype(bool)
        for site_scan in single_subject_sites:
            mask += covars.site_scanner == site_scan
        precombat_features = precombat_features[~mask]
        covars = covars[~mask]
        return covars, precombat_features

    @property
    def calibration_smoothing(self):
        """caliration curve for smoothing surface mesh'"""
        if self._calibration_smoothing is None:
            # Use dictionary based on Freesurfer mris_fwhm values
            y = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,])
            x = np.array([0, 1, 3, 6, 11, 18, 34, 45, 57, 70, 85, 101, 119, 138, 158, 180, 203, 228, 257, 282, 310, 341, 372, 405, 440, 476, 513, 552, 592, 633,])
            # Uncomment to fit a polynom
#             model = np.poly1d(np.polyfit(x, y, 3))
#             x = np.linspace(0, x[-1], x[-1]+1)
#             y = model(x)
            # or uncomment to use homemade function to create calibration curve
#             p = os.path.join(self.data_dir, SMOOTH_CALIB_FILE)
#             coords, faces = nb.freesurfer.io.read_geometry(p)
#             x, y = mt.calibrate_smoothing(coords, faces, start_v=125000, n_iter=300)          
            self._calibration_smoothing = (x, y)
        return self._calibration_smoothing
    
    def clip_data(self, vals, params):
        """ clip data to remove very extreme feature values """
        min_p = float(params['min_percentile'])
        max_p = float(params['max_percentile'])
        num = (vals < min_p).sum() + (vals > max_p).sum()
        vals = np.clip(vals, min_p, max_p)
        return vals, num
    
    def smooth_data(self, feature, fwhm, clipping_params, outliers_file=None):
        """smooth features with given fwhm for all subject and save in new hdf5 file"""
        # create smooth name
        feature_smooth = self.feat.smooth_feat(feature, fwhm)
        # initialise
        neighbours = self.cohort.neighbours
        subject_include = []
        vals_matrix_lh = []
        vals_matrix_rh = []
        if clipping_params!=None:
            print(f'INFO - all: Clip data to remove very extreme values using {clipping_params}')
        for id_sub in self.subject_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            # smooth data only if the feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                # harmonise sulcus data from freesurfer v5 and v6
                if feature == ".on_lh.sulc.mgh":
                    vals_lh = self.correct_sulc_freesurfer(vals_lh, self.cohort.cortex_mask)
                    vals_rh = self.correct_sulc_freesurfer(vals_rh, self.cohort.cortex_mask)
                # clip data to remove outliers vertices
                if clipping_params!=None:
                    with open(os.path.join(self.meld_dir,clipping_params), "r") as f:
                        params = json.loads(f.read())
                        vals_lh, num_lh = self.clip_data(vals_lh, params[feature])
                        vals_rh, num_rh = self.clip_data(vals_rh, params[feature])
                        if (num_lh>0) or (num_rh>0):
                            print(f'WARNING - {id_sub}: {num_lh + num_rh} extremes vertices')
                            header_name = ['subject', 'feature', 'num vertices outliers left', 'num vertices outliers right']
                            if outliers_file!=None:
                                need_header=False
                                if not os.path.isfile(outliers_file):
                                    need_header=True
                                with open(outliers_file, 'a') as f:
                                    writer = csv.writer(f)
                                    if need_header:
                                        writer.writerow(header_name)
                                    writer.writerow([id_sub, feature, num_lh, num_rh])
                vals_matrix_lh.append(vals_lh)
                vals_matrix_rh.append(vals_rh)
                subject_include.append(id_sub)
            else:
                print(f"INFO - {id_sub}: feature {feature} does not exist")
        #if matrix is empty, do nothing
        if not vals_matrix_lh:
            pass
        else:
            # smoothed data if fwhm
            vals_matrix_lh = np.array(vals_matrix_lh)
            vals_matrix_rh = np.array(vals_matrix_rh)
            if fwhm:
                # find number iteration from calibration smoothing
                x, y = self.calibration_smoothing
                idx = (np.abs(y - fwhm)).argmin()
                n_iter = int(np.round(x[idx]))
                print(f"INFO - all : Smoothing with {n_iter} iterations ...")
                vals_matrix_lh = mt.smooth_array(
                    vals_matrix_lh.T, neighbours, n_iter=n_iter, cortex_mask=self.cohort.cortex_mask
                )
                vals_matrix_rh = mt.smooth_array(
                    vals_matrix_rh.T, neighbours, n_iter=n_iter, cortex_mask=self.cohort.cortex_mask
                )
            else:
                print("INFO - all : no smoothing for this feature")
                vals_matrix_lh = vals_matrix_lh.T
                vals_matrix_rh = vals_matrix_rh.T

            smooth_vals_hemis = np.array(
                np.hstack([vals_matrix_lh[self.cohort.cortex_mask].T, vals_matrix_rh[self.cohort.cortex_mask].T])
            )
            # write features in hdf5
            print("INFO - all : saving data")
            self.save_cohort_features(feature_smooth, smooth_vals_hemis, np.array(subject_include))
            return smooth_vals_hemis
        
    def define_atlas(self, atlas=DK_ATLAS_FILE):
        atlas = nb.freesurfer.io.read_annot(os.path.join(self.meld_dir, atlas))
        self.vertex_i = np.array(atlas[0]) - 1000  # subtract 1000 to line up vertex
        self.rois_prop = [
            np.count_nonzero(self.vertex_i == x) for x in set(self.vertex_i)
        ]  # proportion of vertex per rois
        rois = [x.decode("utf8") for x in atlas[2]]  # extract rois label from the atlas
        rois = dict(zip(rois, range(len(rois))))  # extract rois label from the atlas
        rois.pop("unknown")  # roi not part of the cortex
        rois.pop("corpuscallosum")  # roi not part of the cortex
        self.rois = rois

    def get_key(self, dic, val):
        # function to return key for any value in dictionnary
        for key, value in dic.items():
            if val == value:
                return key
        return "WARNING: No key for value {}".format(val)

    def create_features_rois_matrix(self, feature, hemi, save_matrix=False):
        """Compute matrix with average feature values per ROIS for each subject"""
        self.define_atlas()
        matrix = pd.DataFrame()
        for id_sub in self.subject_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            # create a dictionnary to store values for each row of the matrix
            row = {}
            row["ID"] = subj.subject_id
            row["site"] = subj.site_code
            row["scanner"] = subj.scanner
            row["group"] = subj.group
            row["FLAIR"] = subj.has_flair

            # remove rois where more than 25% of vertex are lesional
            rois_s = self.rois.copy()
            if subj.has_lesion == True:
                lesion = subj.load_feature_values(".on_lh.lesion.mgh", hemi)
                rois_lesion = list(self.vertex_i[lesion == 1])
                rois_lesion = [[x, rois_lesion.count(x)] for x in set(rois_lesion) if x != 0]
                for ind, num in rois_lesion:
                    if num / self.rois_prop[ind - 1] * 100 > 30:
                        print("remove {}".format(ind))
                        rois_s.pop(self.get_key(rois_s, ind))

            # compute average feature per rois
            if subj.has_features(feature):
                feat_values = subj.load_feature_values(feature, hemi)
                # correct sulcus values if in mm
                if feature == ".on_lh.sulc.mgh":
                    feat_values = self.correct_sulc_freesurfer(feat_values, self.cohort.cortex_mask)
                # calculate mean thickness & std per ROI
                for roi, r in rois_s.items():
                    row[roi + "." + feature] = np.mean(feat_values[self.vertex_i == r])
            else:
                pass
            #                 print('feature {} does not exist for subject {}'.format(feature,id_sub))
            # add row to matrix
            matrix = matrix.append(pd.DataFrame([row]), ignore_index=True)
        # save matrix
        if save_matrix == True:
            file = os.path.join(self.data_dir, "matrix_QC_{}_wholecohort.csv".format(hemi))
            matrix.to_csv(file)
            print("Matrix with average features/ROIs for all subject can be found at {}".format(file))

        return matrix
    
    def get_outlier_feature(self, feature, hemi):
        """return array of 1 (feature is outlier) and 0 (feature is not outlier) for list of subjects"""
        df = self.create_features_rois_matrix(feature, hemi, save_matrix=True)
#         df = pd.read_csv(os.path.join(self.data_dir, "matrix_QC_{}_wholecohort.csv".format(hemi)), header=0)
        # define if feature is outlier or not
        ids = df.groupby(["site", "scanner"])
        outliers = []
        subjects = []
        for index, row in df.iterrows():
            print(row["ID"])
            subjects.append(row["ID"])
            group = ids.get_group((row["site"], row["scanner"]))
            # warning if not enough subjects per site/scanner
            if len(group.index) <= 6:
                print(
                    "WARNING : only {} subjects in site {} and scanner {}".format(
                        len(group.index), row["site"], row["scanner"]
                    )
                )
            # find upper and lower limit for each ROIS
            lower_lim = group.mean() - 2.698 * group.std()
            upper_lim = group.mean() + 2.698 * group.std()
            # check if subject out of specs
            keys_feat = [key for key in set(df) if feature in key]
            count_out_rois = 0
            for key in keys_feat:
                if (row[key] <= lower_lim[key]) or (row[key] >= upper_lim[key]):
                    count_out_rois += 1
                else:
                    pass
            # decide if feature is outliers based on number or outliers ROIs
            if count_out_rois >= 10:
                outliers.append(1)
            else:
                outliers.append(0)
        return outliers, df[["ID", "FLAIR"]].copy()
    
    def find_outliers(self, features, output_file=None):
        """return list of outliers pre-combat"""
        # Find how many features are outliers per subjec
        tot_out_feat = []
        for feature in features:
            print("Process outlier for feature {}".format(feature))
            out_feat_lh, df = self.get_outlier_feature(feature, "lh")
            out_feat_rh, _ = self.get_outlier_feature(feature, "rh")
            tot_out_feat.append(out_feat_lh)
            tot_out_feat.append(out_feat_rh)
        df["tot_out_feat"] = np.array(tot_out_feat).sum(axis=0)

        # different conditions to define if subject is an outlier
        outliers = df[(df["FLAIR"] == True) & (df["tot_out_feat"] >= 3)]["ID"]
        outliers = outliers.append(df[(df["FLAIR"] == False) & (df["tot_out_feat"] >= 2)]["ID"])
        # save outliers
        if output_file is not None:
            file_path = os.path.join(self.data_dir, output_file)
            print("INFO: list of outliers saved at {}".format(file_path))
            outliers.to_csv(file_path, index=False)

        return outliers
    
    def compute_mean_std_controls(self, feature, cohort, asym=False, params_norm=None):
        """retrieve controls from given cohort, intra-normalise feature and return mean and std for inter-normalisation"""
        controls_ids = cohort.get_subject_ids(group="control")
        # Give warning if list of controls empty
        if len(controls_ids) == 0:
            print("WARNING: there is no controls in this cohort to do inter-normalisation")
        vals_array = []
        included_subj = []
        for id_sub in controls_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=cohort)
            # append data to compute mean and std if feature exist
            if subj.has_features(feature):
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
                # intra subject normalisation asym
                intra_norm = np.array(self.normalise(vals))
                # Calculate asymmetry
                if asym == True:
                    intra_norm = self.compute_asym(intra_norm)
                    names_save = [f'mean.asym',f'std.asym']
                else:
                    names_save = [f'mean',f'std']   
                vals_array.append(intra_norm)
                included_subj.append(id_sub)
            else:
                pass
        print("INFO: Compute mean and std from {} controls".format(len(included_subj)))
        # get mean and std from controls
        params = {}
        params[names_save[0]] = np.mean(vals_array, axis=0)
        params[names_save[1]] = np.std(vals_array, axis=0)
        # save parameters in hdf5
        if params_norm!=None:
            self.save_norm_combat_parameters(feature, params, params_norm)
        return params[names_save[0]], params[names_save[1]]
    
    def normalise(self, data):
        if len(data.shape) == 1:
            data[:, np.newaxis]
        mean_intra = np.mean(data, axis=0)
        std_intra = np.std(data, axis=0)
        intra_norm = (data - mean_intra) / std_intra
        return intra_norm

    def compute_asym(self, intra_norm):
        intra_lh = intra_norm[: int(len(intra_norm) / 2)]
        intra_rh = intra_norm[int(len(intra_norm) / 2) :]
        lh_asym = intra_lh - intra_rh
        rh_asym = intra_rh - intra_lh
        asym = np.hstack([lh_asym, rh_asym])
        return asym
    
    def intra_inter_subject(self, feature, cohort_for_norm=None, params_norm=None):
        """perform intra normalisation (within subject) and
        inter-normalisation (between subjects relative to controls)"""
        feature_norm = self.feat.norm_feat(feature)
        # loop over subjects
        vals_array = []
        included_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        controls_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        for k, id_sub in enumerate(self.subject_ids):
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            if subj.has_features(feature):
                included_subjects[k] = True
                if subj.group == "control":
                    controls_subjects[k] = True
                else:
                    controls_subjects[k] = False
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[self.cohort.cortex_mask], vals_rh[self.cohort.cortex_mask]]))
                # intra subject normalisation asym
                intra_norm = np.array(self.normalise(vals))
                vals_array.append(intra_norm)
            else:
                included_subjects[k] = False
                controls_subjects[k] = False
        print(f"INFO: exclude subjects {np.array(self.subject_ids)[~included_subjects]}")
        if vals_array:
            vals_array = np.array(vals_array)
            # remove exclude subjects
            controls_subjects = np.array(controls_subjects)[included_subjects]
            included_subjects = np.array(self.subject_ids)[included_subjects]
            # normalise by controls
            if cohort_for_norm is not None:
                print("INFO: Use other cohort for normalisation")
                mean_c, std_c = self.compute_mean_std_controls(feature, cohort=cohort_for_norm, 
                                                               params_norm=params_norm)
            else:
                if params_norm is not None:
                    print(f'INFO: Use precomputed normalisation parameters from MELD cohort')
                    params = self.read_norm_combat_parameters(feature, params_norm)
                    mean_c = params['mean']
                    std_c = params['std']
                else : 
                    print(
                        "INFO: Use same cohort for normalisation. Compute mean and std from {} controls".format(
                            controls_subjects.sum()
                        )
                    )
                    mean_c = np.mean(vals_array[controls_subjects], axis=0)
                    std_c = np.std(vals_array[controls_subjects], axis=0)
            vals_combat = (vals_array - mean_c) / std_c
            # save subject
            print("INFO - all: Normalisation finished. Saving data")
            self.save_cohort_features(feature_norm, vals_combat, included_subjects)
        else:
            print('WARNING: No data to normalise')
            pass
        
    def asymmetry_subject(self, feature, cohort_for_norm=None, params_norm=None):
        """perform intra normalisation (within subject) and
        inter-normalisation (between subjects relative to controls) and asymetry between hemispheres"""
        feature_asym = self.feat.asym_feat(feature)
        # loop over subjects
        vals_asym_array = []
        included_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        controls_subjects = np.zeros(len(self.subject_ids), dtype=bool)
        for k, id_sub in enumerate(self.subject_ids):
            # create subject object
            subj = MeldSubject(id_sub, cohort=self.cohort)
            if subj.has_features(feature):
                included_subjects[k] = True
                if subj.group == "control":
                    controls_subjects[k] = True
                else:
                    controls_subjects[k] = False

                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[self.cohort.cortex_mask], vals_rh[self.cohort.cortex_mask]]))
                # intra subject normalisation asym
                intra_norm = np.array(self.normalise(vals))
                # Calculate asymmetry
                vals_asym = self.compute_asym(intra_norm)
                vals_asym_array.append(vals_asym)
            else:
                included_subjects[k] = False
                controls_subjects[k] = False
        print(f"INFO: exclude subjects {np.array(self.subject_ids)[~included_subjects]}")
        if vals_asym_array :
            vals_asym_array = np.array(vals_asym_array)
            # remove exclude subjects
            controls_subjects = np.array(controls_subjects)[included_subjects]
            included_subjects = np.array(self.subject_ids)[included_subjects]
            # normalise by controls
            if cohort_for_norm is not None:
                print("INFO: Use other cohort for normalisation")
                mean_c, std_c = self.compute_mean_std_controls(feature, cohort=cohort_for_norm, asym=True, 
                                                               params_norm=params_norm)
            else:
                if params_norm is not None:
                    print(f'INFO: Use precomputed normalisation parameters from MELD cohort')
                    params = self.read_norm_combat_parameters(feature, params_norm)
                    mean_c = params['mean.asym']
                    std_c = params['std.asym']
                else:
                    print(
                        "INFO: Use same cohort for normalisation. Compute mean and std from {} controls".format(
                            controls_subjects.sum()
                        )
                    )
                    mean_c = np.mean(vals_asym_array[controls_subjects], axis=0)
                    std_c = np.std(vals_asym_array[controls_subjects], axis=0)
            asym_combat = (vals_asym_array - mean_c) / std_c
            # save subject
            print("INFO - all: Asym finished. Saving data")
            self.save_cohort_features(feature_asym, asym_combat, included_subjects)
        else:
            print('WARNING - all: No data to do asym')
            pass
    
    def compute_mean_std(self, feature, cohort):
        """get mean and std of all brain for the given cohort and save parameters"""
        cohort_ids = cohort.get_subject_ids(group="both")
        # Give warning if list of controls empty
        if len(cohort_ids) == 0:
            print("WARNING: there is no subject in this cohort")
        vals_array = []
        included_subj = []
        for id_sub in cohort_ids:
            # create subject object
            subj = MeldSubject(id_sub, cohort=cohort)
            # append data to compute mean and std if feature exist and for FLAIR=0
            if (not subj.has_features(feature)) & (not 'FLAIR' in feature):
                pass 
                print('INFO: feature {} does not exist for subject {}'.format(feature,id_sub))
            else:
                # load feature's value for this subject
                vals_lh = subj.load_feature_values(feature, hemi="lh")
                vals_rh = subj.load_feature_values(feature, hemi="rh")
                vals = np.array(np.hstack([vals_lh[cohort.cortex_mask], vals_rh[cohort.cortex_mask]]))
                vals_array.append(vals)
                included_subj.append(id_sub)                
        print("INFO: Compute mean and std from {} subject".format(len(included_subj)))
        # get mean and std
        vals_array = np.matrix(vals_array)
        mean = (vals_array.flatten()).mean()
        std = (vals_array.flatten()).std()
        # save in json
        data = {}
        data["{}".format(feature)] = {
            "mean": str(mean),
            "std": str(std),
        }
        # create or re-write json file
        file = os.path.join(self.data_dir, self.write_output_file)
        if os.path.isfile(file):
            # open json file and get dictionary
            with open(file, "r") as f:
                x = json.loads(f.read())
            # update dictionary with new dataset version
            x.update(data)
        else:
            x = data
        # save dictionary in json file
        with open(file, "w") as outfile:
            json.dump(x, outfile, indent=4)
        print(f"parameters saved in {file}")

    def curvature_regress(self, feature, curv_feature):
        target_feature = self.feat.regress_feat(feature)
        for si,sub_id in enumerate(self.subject_ids):
            # if si %100==0:
            #     print(si)
            subj = MeldSubject(sub_id, cohort=self.cohort)
            try:
                thickness_lh = subj.load_feature_values(feature, hemi="lh")
                curvature_lh = subj.load_feature_values(curv_feature, hemi="lh")

                thickness_rh = subj.load_feature_values(feature, hemi="rh")
                curvature_rh = subj.load_feature_values(curv_feature, hemi="rh")
                lh_reg= surface_regression(thickness_lh[self.cohort.cortex_mask],
                                    curvature_lh[self.cohort.cortex_mask]
                        )
                rh_reg = surface_regression(thickness_rh[self.cohort.cortex_mask],
                                            curvature_rh[self.cohort.cortex_mask]
                                )
                vals_reg = np.concatenate([lh_reg,rh_reg])
                subj.write_feature_values(target_feature, 
                                        vals_reg, hemis=['lh','rh'],
                                        hdf5_file_root=self.write_output_file)
            except KeyError:
                print(sub_id)
        return

def surface_regression(metric_in, remove):
        from scipy.stats import linregress
        remove_means = np.mean(remove)
        remove_data = remove - remove_means        
        remove_slope = linregress(remove_data, metric_in).slope
        regress_scaled = remove_data * remove_slope
        metric_out = metric_in - regress_scaled
        return metric_out
    
  
class Feature:
    def __init__(self):
        """Class to define feature name"""
        pass

    def raw_feat(self, feature):
        self._raw_feat = feature
        return self._raw_feat

    def smooth_feat(self, feature, smoother=None):
        if smoother != None:
            smooth_part = "sm" + str(int(smoother))
            list_name = feature.split(".")
            new_name = list(chain.from_iterable([list_name[0:-1], [smooth_part, list_name[-1]]]))
            self._smooth_feat = ".".join(new_name)
        else:
            self._smooth_feat = feature
        return self._smooth_feat

    def regress_feat(self, feature):
        split = feature.split(".sm")
        return "".join([split[0], "_regression", ".sm", split[-1]])
    
    def combat_feat(self, feature):
        return "".join([".combat", feature])

    def norm_feat(self, feature):
        self._norm_feat = "".join([".inter_z.intra_z", feature])
        return self._norm_feat

    def asym_feat(self, feature):
        self._asym_feat = "".join([".inter_z.asym.intra_z", feature])
        return self._asym_feat
    
    def norm_GP_feat(self, feature):
        self._norm_GP_feat = "".join([".GP_norm", feature])
        return self._norm_GP_feat
    
    def asym_GP_feat(self, feature):
        self._asym_GP_feat = "".join([".asym", feature])
        return self._asym_GP_feat

    def list_feat(self):
        self._list_feat = [self.smooth, self.combat, self.norm, self.asym]
        return self._list_feat
    