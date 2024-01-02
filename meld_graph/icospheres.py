import os
import numpy as np
import nibabel as nb
from scipy import sparse
import meld_graph.mesh_tools as mt
import torch
from math import pi
import logging


class IcoSpheres:
    """
    Icospheres representation, with functions for loading, downsampling, upsampling, etc.

    Icospheres at each level are stored in self.icospheres[1:7].
    Autoloads & calculates:
        'coords': spherical coordinates
        'faces': triangle faces
        'polar_coords': theta & phi spherical coords
        'edges': all edges
        'adj_mat': sparse adjacency matrix

    Args:
        distance_type (str): 'exact' or 'pseudo'
            exact - edge length and flattened relative angle
            pseudo - relative polar coordinates
        conv_type (str): GMMConv or SpiralConv
    """

    def __init__(
        self,
        icosphere_path="data/icospheres/",
        distance_type="pseudo",
        conv_type="GMMConv",
        **kwargs,
    ):
        # TODO already gets combine_hemis as input, can use that to choose edges file
        self.log = logging.getLogger(__name__)
        self.icosphere_path = icosphere_path
        self.icospheres = {}
        self.conv_type = conv_type
        self.distance_type = distance_type
        self.log.debug(f"Using coord type {self.distance_type}")
        self.load_all_levels()

    def load_all_levels(self):
        for level in np.arange(7) + 1:
            self.load_one_level(level=level)
        return

    def load_one_level(self, level=7):
        self.load_icosphere(level=level)
        self.calculate_neighbours(level=level)
        self.spherical_coords(level=level)
        self.get_exact_edge_attrs(level=level)
        self.calculate_adj_mat(level=level)
        if self.conv_type == "SpiralConv":
            self.create_spirals(level=level)
        elif self.conv_type == "GMMConv":
            if self.distance_type == "pseudo":
                self.calculate_pseudo_edge_attrs(level=level)
        return

    def load_icosphere(self, level=7):
        surf_nb = nb.load(os.path.join(self.icosphere_path, f"ico{level}.surf.gii"))
        self.icospheres[level] = {
            "coords": surf_nb.darrays[0].data,
            "faces": surf_nb.darrays[1].data,
        }
        return

    def calculate_adj_mat(self, level=7):
        surf = self.icospheres[level]

        surf["adj_mat"] = sparse.coo_matrix(
            (
                np.ones(len(surf["edges"]), np.uint8),
                (surf["edges"][:, 0], surf["edges"][:, 1]),
            ),
            shape=(len(surf["coords"]), len(surf["coords"])),
        ).tocsr()
        return

    def calculate_neighbours(self, level=7):
        file_path = os.path.join(self.icosphere_path, f"ico{level}.neighbours.npy")
        if os.path.isfile(file_path):
            self.icospheres[level]["neighbours"] = np.load(file_path, allow_pickle=True)
        else:
            self.icospheres[level]["neighbours"] = np.array(
                self.get_neighbours_from_tris(self.icospheres[level]["faces"]),
                dtype=object,
            )
            np.save(file_path, self.icospheres[level]["neighbours"], allow_pickle=True)

        return

    def spherical_coords(self, level=7):
        self.icospheres[level]["spherical_coords"] = mt.spherical_np(self.icospheres[level]["coords"])[:, 1:]
        self.icospheres[level]["spherical_coords"][:, 0] = self.icospheres[level]["spherical_coords"][:, 0] - pi / 2
        return

    def calculate_pseudo_edge_attrs(self, level=7):
        """pseudo edge attributes, difference between latitude and longitude"""
        file_path = os.path.join(self.icosphere_path, f"ico{level}.pseudo.npy")
        if os.path.isfile(file_path):
            pseudo = np.load(file_path)
        else:
            col = self.icospheres[level]["edges"][:, 0]
            row = self.icospheres[level]["edges"][:, 1]
            pos = self.icospheres[level]["spherical_coords"]
            pseudo = pos[col] - pos[row]
            alpha = pseudo[:, 1]
            tmp = (alpha == 0).nonzero()[0]
            alpha[tmp] = 1e-15
            tmp = (alpha < 0).nonzero()[0]
            alpha[tmp] = np.pi + alpha[tmp]
            alpha = 2 * np.pi + alpha
            alpha = np.remainder(alpha, 2 * np.pi)
            pseudo[:, 1] = alpha
        self.icospheres[level]["pseudo_edge_attr"] = pseudo
        self.icospheres[level]["t_pseudo_edge_attr"] = torch.tensor(
            self.icospheres[level]["pseudo_edge_attr"], dtype=torch.float
        )

        return

    def to(self, device):
        """loads edges, edge vectors and neighbors to device (eg GPU)"""
        for level in self.icospheres.keys():
            self.icospheres[level]["t_edges"] = self.icospheres[level]["t_edges"].to(device)
            if self.conv_type == "SpiralConv":
                self.icospheres[level]["spirals"] = self.icospheres[level]["spirals"].to(device)
            elif self.conv_type == "GMMConv":
                if self.distance_type == "exact":
                    self.icospheres[level]["t_exact_edge_attr"] = self.icospheres[level]["t_exact_edge_attr"].to(device)
                elif self.distance_type == "pseudo":
                    self.icospheres[level]["t_pseudo_edge_attr"] = self.icospheres[level]["t_pseudo_edge_attr"].to(
                        device
                    )

        return

    # helper functions
    def get_edges(self, level=7):
        """returns edges tensor"""
        return self.icospheres[level]["t_edges"]

    def get_edge_vectors(self, level=7):
        if self.distance_type == "pseudo":
            return self.icospheres[level]["t_pseudo_edge_attr"]
        elif self.distance_type == "exact":
            return self.icospheres[level]["t_exact_edge_attr"]

    def get_neighbours_from_tris(self, tris):
        """Get surface neighbours from tris
        Input: tris
        Returns Nested list. Each list corresponds
        to the ordered neighbours for the given vertex"""
        n_vert = np.max(tris) + 1
        neighbours = [[] for i in range(n_vert)]
        for tri in tris:
            neighbours[tri[0]].append([tri[1], tri[2]])
            neighbours[tri[2]].append([tri[0], tri[1]])
            neighbours[tri[1]].append([tri[2], tri[0]])
        # Get unique neighbours
        for k in range(len(neighbours)):
            neighbours[k] = self.sort_neighbours(neighbours[k])
        return neighbours

    def sort_neighbours(self, edges):
        edges = np.vstack(edges)
        n0 = edges[0][0]
        sorted_neighbours = np.zeros(len(edges), dtype=int)
        for e_i in np.arange(len(edges)):
            n0 = edges[:, 1][edges[:, 0] == n0][0]
            sorted_neighbours[e_i] = n0
        return sorted_neighbours

    def findAnglesBetweenTwoVectors1(self, v1s, v2s):
        dot = np.einsum("ijk,ijk->ij", [v1s, v1s, v2s], [v2s, v1s, v2s])
        return np.arccos(dot[0, :] / (np.sqrt(dot[1, :]) * np.sqrt(dot[2, :])))

    def calculate_angles_and_dists(self, vertex, neighbours, coords):
        angles = np.zeros(len(neighbours))
        v1 = coords[neighbours] - coords[vertex]
        v2 = coords[np.roll(neighbours, 1)] - coords[vertex]
        angles = self.findAnglesBetweenTwoVectors1(v1, v2)
        total_angle = angles.sum()
        angles_flattened = 2 * pi * angles.cumsum() / total_angle
        return angles_flattened, np.linalg.norm(v1, axis=1)

    def vertex_attributes(self, surf, vertex):
        neighbours = surf["neighbours"][vertex]
        edges = self.neighbours_to_edges(vertex, neighbours)
        angles, dists = self.calculate_angles_and_dists(vertex, neighbours, surf["coords"])
        # add self edge with almost zero vals
        edge_attrs = np.vstack([[1e-15, 1e-15], np.vstack([angles, dists]).T])
        combined = np.hstack([edges, edge_attrs])
        return combined

    def neighbours_to_edges(self, vertex, neighbours):
        """generate paired ordered list of vertex to neighbours, including self edge"""
        edges = np.vstack(
            [
                [vertex, vertex],
                np.vstack([np.repeat(vertex, len(neighbours)), neighbours]).T,
            ]
        )
        return edges

    def get_exact_edge_attrs(self, level=7):
        file_path = os.path.join(self.icosphere_path, f"ico{level}.edges_and_attrs.npy")
        if os.path.isfile(file_path):
            edges_attrs = np.load(file_path)
        else:
            edges_attrs = self.calculate_exact_edge_attrs(level=level)
            np.save(file_path, edges_attrs)
        self.icospheres[level]["edges"] = edges_attrs[:, :2].astype(int)
        self.icospheres[level]["exact_edge_attr"] = edges_attrs[:, 2:]
        # add tensors needed for model
        self.icospheres[level]["t_edges"] = (
            torch.tensor(self.icospheres[level]["edges"], dtype=torch.long).t().contiguous()
        )
        self.icospheres[level]["t_exact_edge_attr"] = torch.tensor(
            self.icospheres[level]["exact_edge_attr"], dtype=torch.float
        )
        return

    def calculate_exact_edge_attrs(self, level=7):
        surf = self.icospheres[level]
        n_vert = len(surf["coords"])
        all_edge_attrs = []
        for v in np.arange(n_vert):
            edge_attrs = self.vertex_attributes(surf, v)
            all_edge_attrs.append(edge_attrs)
        all_edge_attrs = np.vstack(all_edge_attrs)
        return all_edge_attrs

    def get_neighbours(self, level=7):
        """return 7*n_vertex array of neighbours, with self neighbours
        and repeated self index if only 5 neighbours"""
        if "t_neighbours" not in self.icospheres[level].keys():
            self.icospheres[level]["t_neighbours"] = np.tile(np.arange(len(self.icospheres[level]["coords"])), (7, 1)).T
            for ni, n in enumerate(self.icospheres[level]["neighbours"]):
                self.icospheres[level]["t_neighbours"][ni, -len(n) :] = n
            self.icospheres[level]["t_neighbours"] = torch.tensor(
                self.icospheres[level]["t_neighbours"], dtype=torch.long
            )
        return self.icospheres[level]["t_neighbours"]

    def get_downsample(self, target_level=6):
        """return 7*n_vertex array of neighbours, with self neighbours
        and repeated self index if only 5 neighbours"""
        # get neighbours from level above, but restrict to length of lower level mesh
        # used for pooling operations
        if "t_downsample" not in self.icospheres[target_level].keys():
            source_level = target_level + 1
            n_target_vertices = len(self.icospheres[target_level]["coords"])
            self.icospheres[target_level]["t_downsample"] = self.get_neighbours(level=source_level)[:n_target_vertices]
        return self.icospheres[target_level]["t_downsample"]

    def get_upsample(self, target_level=7):
        """provide edges new vertices in mesh upsampled to the target level
        returns array (len(new_level)-len(old_level),2)
        with 2 indices of vertices old level for the new vertex in the new level.
        Row zero describes the vertices for the first new_level vertex
        """
        if target_level == 1:
            print(
                "Trying to upsample to the lowest resolution mesh.",
                "A coarser version of this mesh doesn't exist.",
                "Double check you're using target_level correctly",
            )
            return None
        if "t_upsample" not in self.icospheres[target_level].keys():
            n_vert_down = len(self.icospheres[target_level - 1]["coords"])
            n_vert_up = len(self.icospheres[target_level]["coords"])
            neighbours_to_explore = self.get_neighbours(level=target_level)[n_vert_down:]
            neighbours_to_explore = neighbours_to_explore[neighbours_to_explore < n_vert_down]
            self.icospheres[target_level]["t_upsample"] = neighbours_to_explore.reshape(n_vert_up - n_vert_down, 2)
        return self.icospheres[target_level]["t_upsample"]

    def create_spirals(self, level=7):
        file_path = os.path.join(self.icosphere_path, f"ico{level}.spirals.npy")
        if os.path.isfile(file_path):
            spirals = np.load(file_path)
        else:
            spirals = self.calculate_spirals(level=level)
            np.save(file_path, spirals)
        self.icospheres[level]["spirals"] = torch.tensor(spirals, dtype=torch.long)
        return

    def get_spirals(self, level=7):
        if "spirals" not in self.icospheres[level].keys():
            print(
                "ERROR: Class not initialised with spirals",
                "Either reset convtype or run icos.create_spirals(level=7)",
            )
        return self.icospheres[level]["spirals"]

    def calculate_spirals(self, level=7, size=20):
        """precalculate spinal kernels"""
        neighbours = self.icospheres[level]["neighbours"]
        n_vertices = len(neighbours)
        spirals = np.zeros((n_vertices, size))
        for v in np.arange(n_vertices):
            spirals[v] = self.get_spiral_for_vertex(neighbours, vertex=v, size=size)
        return spirals

    def get_spiral_for_vertex(self, neighbours, vertex=0, size=10):
        """create spiral convolution"""
        vertex_neighbours = neighbours[vertex]
        spiral = [vertex]
        spiral.extend(list(vertex_neighbours))
        # next level starts with neighbour of first and last not vertex
        k = -1
        old_neighbours = vertex_neighbours
        while len(spiral) < size:
            v_start = spiral[-1]
            index_for_rolling = np.where(old_neighbours == v_start)[0][0]
            old_neighbours = np.roll(old_neighbours, len(old_neighbours) - 1 - index_for_rolling)
            new_center_v = old_neighbours[0]
            new_neighbours = neighbours[new_center_v]
            index_for_rolling = np.where(new_neighbours == v_start)[0][0]
            new_neighbours = np.roll(new_neighbours, len(new_neighbours) - 1 - index_for_rolling)
            # stop vertex is next in spiral
            stop_vertex = spiral[np.where(spiral == new_center_v)[0][0] + 1]
            stop_index = np.where(new_neighbours == stop_vertex)[0][0]
            spiral.extend(list(new_neighbours[:stop_index]))
            old_neighbours = new_neighbours

        return np.array(spiral[:size])
