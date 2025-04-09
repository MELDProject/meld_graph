#Contains MeldCohort and MeldSubject classes

from contextlib import contextmanager
from meld_graph.paths import (
    DEMOGRAPHIC_FEATURES_FILE,
    CORTEX_LABEL_FILE,
    SURFACE_FILE,
    DEFAULT_HDF5_FILE_ROOT,
    NVERT,
    BASE_PATH,
    MELD_PARAMS_PATH,
    MELD_DATA_PATH,
)
import pandas as pd
import numpy as np
import nibabel as nb
import os
import h5py
import sys
import glob
import logging
import meld_graph.mesh_tools as mt
import scipy


class MeldCohort:
    """Class to define cohort-level parameters such as subject ids, mesh"""
    def __init__(self, hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset=None, data_dir=BASE_PATH, meld_dir=MELD_PARAMS_PATH):
        self.data_dir = data_dir
        self.meld_dir = meld_dir
        self.hdf5_file_root = hdf5_file_root
        self.dataset = dataset
        self.log = logging.getLogger(__name__)

        # class properties (readonly attributes):
        # full_feature_list: list of features available in this cohort
        self._full_feature_list = None

        # surface information known to MeldCohort
        # cortex_label: information about which nodes are cortex
        self._cortex_label = None
        self._cortex_mask = None
        # coords: spherical 2D coordinates
        self._coords = None
        # surf: inflated mesh, surface vertices and triangles
        self._surf = None
        # surf_partial: partially inflated mesh, surface vertices and triangles
        self._surf_partial = None
        # surf_area: surface area for each triangle
        self._surf_area = None
        # adj_mat: sparse adjacency matrix for all vertices
        self._adj_mat = None
        # lobes: labels for cortical lobes
        self._lobes = None
        # neighbours: list of neighbours for each vertex
        self._neighbours = None

    @property
    def full_feature_list(self):
        """list of features available in this cohort"""
        if self._full_feature_list is None:
            self._full_feature_list = []
            subject_ids = self.get_subject_ids()
            # get union of all features from subjects in this cohort
            features = set()
            for subj in subject_ids:
                features = features.union(MeldSubject(subj, self).get_feature_list().copy())
            self._full_feature_list = sorted(list(features))
            self.log.info(f"full_feature_list: {self._full_feature_list}")
        return self._full_feature_list

    @property
    def cortex_label(self):
        if self._cortex_label is None:
            p = os.path.join(self.meld_dir, CORTEX_LABEL_FILE)
            self._cortex_label = np.sort(nb.freesurfer.io.read_label(p))
        return self._cortex_label

    @property
    def cortex_mask(self):
        if self._cortex_mask is None:

            self._cortex_mask = np.zeros(NVERT, dtype=bool)
            self._cortex_mask[self.cortex_label] = True
        return self._cortex_mask

    @property
    def surf_area(self):
        if self._surf_area is None:
            p = os.path.join(self.meld_dir, "fsaverage_sym/surf/lh.area")
            self._surf_area = nb.freesurfer.read_morph_data(p)
        return self._surf_area

    @property
    def surf(self):
        """inflated surface, dict with 'faces' and 'coords'"""
        if self._surf is None:
            p = os.path.join(self.meld_dir, "fsaverage_sym", "surf", "lh.inflated")
            self._surf = mt.load_mesh_geometry(p)
        return self._surf

    @property
    def surf_partial(self):
        """partially inflated surface, dict with 'faces' and 'coords'"""
        if self._surf_partial is None:
            p = os.path.join(self.meld_dir, "fsaverage_sym", "surf", "lh.partial_inflated")
            vertices, faces = nb.freesurfer.io.read_geometry(p)
            self._surf_partial = {"faces": faces, "coords": vertices}
        return self._surf_partial

    @property
    def adj_mat(self):
        if self._adj_mat is None:
            all_edges = np.vstack(
                [self.surf["faces"][:, :2], self.surf["faces"][:, 1:3], self.surf["faces"][:, [2, 0]]]
            )
            self._adj_mat = scipy.sparse.coo_matrix(
                (np.ones(len(all_edges), np.uint8), (all_edges[:, 0], all_edges[:, 1])),
                shape=(len(self.surf["coords"]), len(self.surf["coords"])),
            ).tocsr()
        return self._adj_mat

    @property
    def neighbours(self):
        if self._neighbours is None:
            self._neighbours = mt.get_neighbours_from_tris(self.surf["faces"])
        return self._neighbours

    @property
    def lobes(self):
        if self._lobes is None:
            p = os.path.join(self.meld_dir, "fsaverage_sym/label/lh.lobes.annot")
            self._lobes = nb.freesurfer.read_annot(p)
        return self._lobes

    @property
    def coords(self):
        if self._coords is None:
            surf = mt.load_mesh_geometry(os.path.join(self.meld_dir, SURFACE_FILE))
            # spherical 2D coordinates. ignore radius
        #    spherical_coords = mt.spherical_np(surf["coords"])[:, 1:]
            # surf_coords_norm = (surf['coords']-np.min(surf['coords'],axis=0))/(np.max(surf['coords'],axis=0)-np.min(surf['coords'],axis=0))
         #   norm_coords = (spherical_coords - np.min(spherical_coords, axis=0)) / (
          #      np.max(spherical_coords, axis=0) - np.min(spherical_coords, axis=0)
          #  )
            # round to have around 1500 unique coordinates
          #  rounded_norm_coords = np.round(norm_coords * 5, 1) / 5
            self._coords = surf["coords"] #rounded_norm_coords
        return self._coords

    def read_subject_ids_from_dataset(self):
        """Read subject ids from the dataset csv file.
        Returns subject_ids, trainval_ids, test_ids"""
        assert self.dataset is not None, "please set a valid dataset csv file"
        df = pd.read_csv(os.path.join(self.data_dir, self.dataset))
        subject_ids = list(df.subject_id)
        trainval_ids = list(df[df.split == "trainval"].subject_id)
        test_ids = list(df[df.split == "test"].subject_id)
        return subject_ids, trainval_ids, test_ids

    def get_sites(self):
        """get all valid site codes that exist on this system"""
        sites = []
        for f in glob.glob(os.path.join(self.data_dir, "MELD_*")):
            if os.path.isdir(f):
                sites.append(f.split("_")[-1])
        return sites

    @contextmanager
    def _site_hdf5(self, site_code, group, write=False, hdf5_file_root=None):
        """
        Hdf5 file handle for specified site_code and group (patient or control).

        This function is to be used in a context block as follows:
        ```
            with cohort._site_hdf5('H1', 'patient') as f:
                # read information from f
                pass
            # f is automatically closed outside of the `with` block
        ```

        Args:
            site_code: hospital site code, e.g. 'H1'
            group: 'patient' or 'control'
            write (optional): flag to open hdf5 file with writing permissions, or to create
                the hdf5 if it does not exist.

        Yields: a pointer to the opened hdf5 file.
        """
        if hdf5_file_root is None:
            hdf5_file_root = self.hdf5_file_root

        p = os.path.join(self.data_dir, f"MELD_{site_code}", hdf5_file_root.format(site_code=site_code, group=group))
        # open existing file or create new one
        if os.path.isfile(p) and not write:
            f = h5py.File(p, "r")
        elif os.path.isfile(p) and write:
            f = h5py.File(p, "r+")
        elif not os.path.isfile(p) and write:
            f = h5py.File(p, "a")
        else:
            f = None
        try:
            yield f
        finally:
            if f is not None:
                f.close()

    def get_subject_ids(self, **kwargs):
        """Output list of subject_ids.

        List can be filtered by sites (given as list of site_codes, e.g. 'H2'),
        groups (patient / control / both), features (subject_features_to_exclude),


        Sites are given as a list of site_codes (e.g. 'H2').
        Optionally filter subjects by group (patient or control).
        If self.dataset is not none, restrict subjects to subjects in dataset csv file.
        subject_features_to_exclude: exclude subjects that dont have this feature

        Args:
            site_codes (list of str): hospital site codes, e.g. ['H1'].
            group (str): 'patient', 'control', or 'both'.
            subject_features_to_exclude (list of str): exclude subjects that dont have this feature
            subject_features_to_include (list of str): exclude subjects that have this feature
            scanners (list of str): list of scanners to include
            lesional_only (bool): filter out lesion negative patients

        Returns:
            subject_ids: the list of subject ids
        """
        # parse kwargs:
        # get groups
        if kwargs.get("group", "both") == "both":
            groups = ["patient", "control"]
        else:
            groups = [kwargs.get("group", "both")]
        # get sites
        site_codes = kwargs.get("site_codes", self.get_sites())
        if isinstance(site_codes, str):
            site_codes = [site_codes]
        # get scanners
        scanners = kwargs.get("scanners", ["3T", "15T", "XT"])
        if not isinstance(scanners, list):
            scanners = [scanners]

        lesional_only = kwargs.get("lesional_only", True)
        subject_features_to_exclude = kwargs.get("subject_features_to_exclude", [""])
        subject_features_to_include = kwargs.get("subject_features_to_include", [""])

        # get subjects for specified groups and sites
        subject_ids = []
        for site_code in site_codes:
            for group in groups:
                with self._site_hdf5(site_code, group) as f:
                    if f is None:
                        continue
                    cur_scanners = f[site_code].keys()
                    for scanner in cur_scanners:
                        subject_ids += list(f[os.path.join(site_code, scanner, group)].keys())

        self.log.info(f"total number of subjects: {len(subject_ids)}")

        # restrict to ids in dataset (if specified)
        if self.dataset is not None:
            subjects_in_dataset, _, _ = self.read_subject_ids_from_dataset()
            subject_ids = list(np.array(subject_ids)[np.in1d(subject_ids, subjects_in_dataset)])
            self.log.info(
                f"total number of subjects after restricting to subjects from {self.dataset}: {len(subject_ids)}"
            )

        # get list of features that is used to filter subjects
        # e.g. use this to filter subjects without FLAIR features
        _, required_subject_features = self._filter_features(
            subject_features_to_exclude,
            return_excluded=True,
        )
        self.log.debug("selecting subjects that have features: {}".format(required_subject_features))

        # get list of features that determine whether to exclude subjects
        # e.g. use this to filter subjects with FLAIR features
        _, undesired_subject_features = self._filter_features(
            subject_features_to_include,
            return_excluded=True,
        )
        self.log.debug("selecting subjects that don't have features: {}".format(undesired_subject_features))

        # filter ids by scanner, features and whether they have lesions.
        filtered_subject_ids = []
        for subject_id in subject_ids:
            subj = MeldSubject(subject_id, self)
            # check scanner
            if subj.scanner not in scanners:
                continue
            # check required features
            if not subj.has_features(required_subject_features):
                continue
            # check undesired features
            if subj.has_features(undesired_subject_features) and len(undesired_subject_features) > 0:
                continue
            # check lesion mask presence
            if lesional_only and subj.is_patient and not subj.has_lesion():
                continue
            # subject has passed all filters, add to list
            filtered_subject_ids.append(subject_id)

        self.log.info(
            f"total number after filtering by scanner {scanners}, features, lesional_only {lesional_only}: {len(filtered_subject_ids)}"
        )
        return filtered_subject_ids

    def get_features(self, features_to_exclude=[""]):
        """
        get filtered list of features.
        """
        # get list of all features that we want to train models on
        # if a subject does not have a feature, 0 is returned for this feature during dataset creation
        features = self._filter_features(features_to_exclude=features_to_exclude)
        self.log.debug("features that will be loaded in train/test datasets: {}".format(features))
        return features

    def _filter_features(self, features_to_exclude, return_excluded=False):
        """Return a list of features, with features_to_exclude removed.

        Args:
            features_to_exclude (list of str): list of features that should be excluded,
                NB 'FLAIR' will exclude all FLAIR features but all other features must be exact matches
            return_excluded (bool): if True, return list of excluded features.

        Returns:
            tuple:
                features: the list of features with appropriate features excluded.
                excluded_features: list of all excluded features. Only returned, if return_exluded is specified.
        """

        all_features = self.full_feature_list.copy()
        excludable_features = []
        filtered_features = self.full_feature_list.copy()
        for feature in self.full_feature_list.copy():
            for exclude in features_to_exclude:
                if exclude == "":
                    pass
                elif exclude == "FLAIR":
                    if exclude in feature:
                        filtered_features.remove(feature)
                        excludable_features.append(feature)

                elif feature == exclude:
                    if exclude in self.full_feature_list:  # only remove if still in list
                        filtered_features.remove(feature)
                        excludable_features.append(feature)
        if return_excluded:
            return filtered_features, excludable_features
        else:
            return filtered_features

    def split_hemispheres(self, input_data):
        """
        split vector of cortex-masked data back into 2 full overlays,
        including zeros for medial wall

        Returns:
            hemisphere_data: dictionary with keys "left" and "right".
        """
        # make sure that input_data has expected format
        assert len(input_data) == 2 * len(self.cortex_label)
        # split data in two hemispheres
        hemisphere_data = {}
        for i, hemi in enumerate(["left", "right"]):
            feature_data = np.zeros((NVERT,) + input_data.shape[1:])
            feature_data[self.cortex_label] = input_data[i * len(self.cortex_label) : (i + 1) * len(self.cortex_label)]
            hemisphere_data[hemi] = feature_data
        return hemisphere_data


class MeldSubject:
    """
    individual patient from meld cohort, can read subject data and other info
    """

    def __init__(self, subject_id, cohort):
        self.subject_id = subject_id
        self.cohort = cohort
        self.log = logging.getLogger(__name__)
        # unseeded rng for generating random numbers
        self.rng = np.random.default_rng()

    @property
    def scanner(self):
        # Note: no need to specify scanner strength with MELD Graph pipeline, but still need it to be compatible with previous MELD FCD dataset
        scanner = self.get_demographic_features('Scanner')
        if scanner is None:
            scanner="XT" #no need to specify 
        if scanner in ("15T" , "1.5T" , "15t" , "1.5t" ):
            scanner="15T" # to be compatible with old way
        elif scanner in ("3T" , "3t" ):
            scanner="3T" # to be compatible with old way
        else:
            scanner="XT" #no need to specify 
        return scanner

    @property
    def group(self):
        group = self.get_demographic_features('Group')
        if (group != "patient") and (group != "control") :
            print(
                f"Error: incorrect group for {self.subject_id}. Unable to determine if patient or control."
            )
            sys.exit()
        return group

    @property
    def site_code(self):
        site_code = self.get_demographic_features('Harmo code')
        return site_code

    def surf_dir_path(self, hemi):
        """return path to features dir (surf_dir)"""
        return os.path.join(self.site_code, self.scanner, self.group, self.subject_id, hemi)

    def find_path(self, name):
        """ Find the first object with the subject id in the hdf5"""
        if self.subject_id in name:
            return name    
    
    
    @property
    def is_patient(self):
        return self.group == "patient"

    @property
    def has_flair(self):
        return "FLAIR" in " ".join(self.get_feature_list())

    def has_lesion(self):
        return self.get_lesion_hemisphere() in ["lh", "rh"]

    def get_lesion_hemisphere(self):
        """
        return 'lh', 'rh', or None
        """
        if not self.is_patient:
            return None

        with self.cohort._site_hdf5(self.site_code, self.group) as f:
            surf_dir_lh = f[os.path.join(self.site_code, f[self.site_code].visit(self.find_path), "lh")]
            if ".on_lh.lesion.mgh" in surf_dir_lh.keys():
                return "lh"
            surf_dir_rh = f[os.path.join(self.site_code, f[self.site_code].visit(self.find_path), "rh")]
            if ".on_lh.lesion.mgh" in surf_dir_rh.keys():
                return "rh"
        return None

    def has_features(self, features):
        missing_features = np.setdiff1d(features, self.get_feature_list())
        return len(missing_features) == 0

    def get_feature_list(self, hemi="lh"):
        """Outputs a list of the features a participant has for each hemisphere"""
        with self.cohort._site_hdf5(self.site_code, self.group) as f:
            surf_dir_path = os.path.join(self.site_code, f[self.site_code].visit(self.find_path), hemi)
            keys =  list(f[surf_dir_path].keys())
            # remove lesion and boundaries from list of features
            if ".on_lh.lesion.mgh" in keys:
                keys.remove(".on_lh.lesion.mgh")
            if ".on_lh.boundary_zone.mgh" in keys:
                keys.remove(".on_lh.boundary_zone.mgh")
        return keys

    def get_demographic_features(
        self, feature_names, csv_file=DEMOGRAPHIC_FEATURES_FILE, normalize=False, default=None
    ):
        """
        Read demographic features from csv file. Features are given as (partial) column titles

        Args:
            feature_names: list of partial column titles of features that should be returned
            csv_path: csv file containing demographics information.
                can be raw participants file or qc-ed values.
                "{site_code}" is replaced with current site_code.
            normalize: implemented for "Age of Onset" and "Duration"
            default: default value to be used when subject does not exist.
                Either "random" (which will choose a random value from the current
                demographics feature column) or any other value which will be used
                as default value.
        Returns:
            list of features, matching structure of feature_names
        """
        # csv_path = os.path.join(MELD_DATA_PATH, csv_file)
        return_single = False
        if isinstance(feature_names, str):
            return_single = True
            feature_names = [feature_names]
        df = pd.read_csv(csv_file, header=0, encoding="latin")
        # get index column
        id_col = None
        for col in df.keys():
            if "ID" in col:
                id_col = col
        # ensure that found an index column
        if id_col is None:
            self.log.warning("No ID column found in file, please check the csv file")

            return None
        df = df.set_index(id_col)
        # find desired demographic features
        features = []
        for desired_name in feature_names:
            matched_name = None
            for col in df.keys():
                if desired_name in col:
                    if matched_name is not None:
                        # already found another matching col
                        self.log.warning(
                            f"Multiple columns matching {desired_name} found ({matched_name}, {col}), please make search more specific"
                        )
                        return None
                    matched_name = col
            # ensure that found necessary data
            if matched_name is None:

                if "urfer" in desired_name:
                    matched_name = "Freesurfer_nul"
                elif "Scanner" in desired_name:
                    return None
                else:
                    self.log.warning(f"Unable to find column matching {desired_name}, please double check for typos")
                    return None

            # read feature
            # if subject does not exists, add None
            if self.subject_id in df.index:
                if matched_name == "Freesurfer_nul":
                    feature = "5.3"
                else:
                    feature = df.loc[self.subject_id][matched_name]
                if normalize:
                    if matched_name == "Age of onset":
                        feature = np.log(feature + 1)
                        feature = feature / df[matched_name].max()
                    elif matched_name == "Duration":
                        feature = (feature - df[matched_name].min()) / (df[matched_name].max() - df[matched_name].min())
                    else:
                        self.log.info(f"demographic feature normalisation not implemented for feature {matched_name}")
            elif default == "random":
                # unseeded rng for generating random numbers
                rng = np.random.default_rng()
                feature = np.clip(np.random.normal(0, 0.1) + rng.choice(df[matched_name]), 0, 1)
            else:
                feature = default
            features.append(feature)
        if return_single:
            return features[0]
        return features

    def load_feature_values(self, feature, hemi="lh"):
        """
        Load and return values of specified feature.
        """
        feature_values = np.zeros(NVERT, dtype=np.float32)
        # read data from hdf5
        with self.cohort._site_hdf5(self.site_code, self.group) as f:
            surf_dir = f[os.path.join(self.site_code, f[self.site_code].visit(self.find_path), hemi)]
            if feature in surf_dir.keys():
                feature_values[:] = surf_dir[feature][:]
            else:
                self.log.debug(f"missing feature: {feature} set to zero")
        return feature_values

    def load_feature_lesion_data(self, features, hemi="lh", features_to_ignore=[]):
        """
        Load all patient's data into memory

        Args:
            features: list of features to be loaded
            hemi: 'lh' or 'rh'
            features_to_ignore: list of features that should be replaced with 0 upon loading

        Returns:
            feature_data, label

        """
        # load all features
        feature_values = []
        for feature in features:
            if feature in features_to_ignore:
                # append zeros for features_to_ignore
                feature_values.append(np.zeros(NVERT, dtype=np.float32))
            else:
                # read feature_values
                feature_values.append(self.load_feature_values(feature, hemi=hemi))
        feature_values = np.stack(feature_values, axis=-1)
        # load lesion data
        lesion_values = np.ceil(self.load_feature_values(".on_lh.lesion.mgh", hemi=hemi)).astype(int)

        return feature_values, lesion_values

    def load_boundary_zone(self, max_distance=40, feat_name=".on_lh.boundary_zone.mgh"):
        """
        load and return boundary zone mask

        max_distance - distance from lesion mask to extend boundary zone in mm
                       30 for training exclusion, 20 for sensitivity testing
        """
        cortex_mask = self.cohort.cortex_mask
        boundary_zones = np.zeros(2 * sum(cortex_mask)).astype(float)
        hemi = self.get_lesion_hemisphere()
        for k, h in enumerate(["lh", "rh"]):
            if hemi == h:
                bz = self.load_feature_values(feat_name, hemi=hemi)
                if max_distance is not None:
                    bz = bz < max_distance
                boundary_zones[k * sum(cortex_mask) : (k + 1) * sum(cortex_mask)] = bz[cortex_mask]
            else:
                bz = np.zeros(len(cortex_mask))
                boundary_zones[k * sum(cortex_mask) : (k + 1) * sum(cortex_mask)] = bz[cortex_mask]

        return boundary_zones

    def get_histology(self):
        """
        get histological classification from cleaned up demographics files
        """
        histology = self.get_demographic_features("Histo")
        return histology

    # TODO write test
    def write_feature_values(self, feature, feature_values, hemis=["lh", "rh"], hdf5_file=None, hdf5_file_root=None):
        """
        write feature to subject's hdf5.

        Args:
            feature: name of the feature
            feature_values: feature values to be written to the hdf5
            hemis: hemispheres that should be written. If only one hemisphere is given,
            it is assumed that all values given with feature_values belong to this hemisphere.
            hdf5_file: uses self.cohort._site_hdf5 by default, but another filename can be specified,
                e.g. to write predicted lesions to another hdf5
            hdf5_file_root: optional to specify a different root from baseline, if writing to a new file
        """
        # check that feature_values have expected length
        if hdf5_file_root is None:
            hdf5_file_root = self.cohort.hdf5_file_root
        assert len(feature_values) == sum(self.cohort.cortex_mask) * len(hemis)
        n_vert_cortex = sum(self.cohort.cortex_mask)
        # open hdf5 file
        if hdf5_file is not None:
            if not os.path.isfile(hdf5_file):
                hdf5_file_context = h5py.File(hdf5_file, "a")
            else:
                hdf5_file_context = h5py.File(hdf5_file, "r+")
        else:
            hdf5_file_context = self.cohort._site_hdf5(
                self.site_code, self.group, write=True, hdf5_file_root=hdf5_file_root
            )
        with hdf5_file_context as f:

            for i, hemi in enumerate(hemis):
                group = f.require_group(self.surf_dir_path(hemi))
                hemi_data = np.zeros(NVERT)
                hemi_data[self.cohort.cortex_mask] = feature_values[i * n_vert_cortex : (i + 1) * n_vert_cortex]
                dset = group.require_dataset(
                    feature, shape=(NVERT,), dtype="float32", compression="gzip", compression_opts=9
                )
                dset[:] = hemi_data

    def delete(self, f, feat):
        print("delete")
        del f[feat]

    def get_lesion_area(self):
        """
        calculate lesion area as the proportion of the hemisphere that is lesion.

        Returns:
            lesion_area, lesion_hemisphere, lesion_lobe
        """
        hemi = self.get_lesion_hemisphere()
        lobes_i, _, lobes_labels = self.cohort.lobes
        if hemi is not None:
            lesion = self.load_feature_values(".on_lh.lesion.mgh", hemi=hemi).astype(bool)
            total_area = np.sum(self.cohort.surf_area[self.cohort.cortex_mask])
            lesion_area = np.sum(self.cohort.surf_area[lesion]) / total_area
            locations = np.unique(lobes_i[lesion], return_counts=True)
            lobe = lobes_labels[locations[0][np.argmax(locations[1])]].decode()
            # set cingulate and insula to second most prevelant
            if lobe in ["cingulate", "insula"]:
                try:
                    lobe = lobes_labels[locations[0][np.argsort(locations[1])[-2]]].decode()
                except IndexError:
                    # one fail case was cingulate next to frontal, so frontal
                    lobe = "frontal"
        else:
            lesion_area = float("NaN")
            hemi = "NaN"
            lobe = "NaN"
        return lesion_area, hemi, lobe