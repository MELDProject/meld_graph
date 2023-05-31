### functions needed to create transformations on spherical mesh for augmentation.
### These include a warp, spins and reindexing the coordinates.

from meld_graph import icospheres
from meld_graph.icospheres import IcoSpheres
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group


def warp_mesh(low_res_ico, warp_fraction=2):
    """adjust the locations of vertices in a low-res version of the mesh"""
    non_self = low_res_ico["edges"][:, 0] != low_res_ico["edges"][:, 1]
    edge_shift = np.linalg.norm(low_res_ico["exact_edge_attr"], axis=1)[non_self]
    max_movement = np.min(edge_shift) / warp_fraction
    surf = low_res_ico["coords"].copy()
    adjustment = np.random.uniform(-max_movement, max_movement, size=(len(surf), 3))
    over = np.linalg.norm(adjustment, axis=1) > max_movement
    adjustment[over] = (max_movement * (adjustment[over]).T / np.linalg.norm(adjustment[over], axis=1)).T
    surf += adjustment
    surf = (surf.T / np.linalg.norm(surf, axis=1)).T
    return surf


def upsample_mesh(warped_coords, lower_res, higher_res):
    """upsample the warped coordinates to make a high-res mesh that
    has large-scale deformations"""
    small_surf_size = len(lower_res["coords"])
    big_surf_size = len(higher_res["coords"])
    big_neighbours = higher_res["neighbours"]
    new_coords = np.zeros((big_surf_size, 3))
    new_coords[:small_surf_size] = warped_coords
    for v in np.arange(big_surf_size)[small_surf_size:]:
        my_ns = big_neighbours[v]
        my_ns = my_ns[my_ns < small_surf_size]
        new_coords[v] = np.mean(warped_coords[my_ns], axis=0)
    new_coords = (new_coords.T / np.linalg.norm(new_coords, axis=1)).T
    return 100 * new_coords


def barycentric_coordinates_matrix(p, tri):
    """solve to return coordinates as barycentric from 3 vertices of triangle.
    Use outputs for linear interpolation"""
    a = (
        np.square(tri[:, 0, 0] - tri[:, 2, 0])
        + np.square(tri[:, 0, 1] - tri[:, 2, 1])
        + np.square(tri[:, 0, 2] - tri[:, 2, 2])
    )
    b = (
        (tri[:, 1, 0] - tri[:, 2, 0]) * (tri[:, 0, 0] - tri[:, 2, 0])
        + (tri[:, 1, 1] - tri[:, 2, 1]) * (tri[:, 0, 1] - tri[:, 2, 1])
        + (tri[:, 1, 2] - tri[:, 2, 2]) * (tri[:, 0, 2] - tri[:, 2, 2])
    )
    c = b
    d = (
        np.square(tri[:, 1, 0] - tri[:, 2, 0])
        + np.square(tri[:, 1, 1] - tri[:, 2, 1])
        + np.square(tri[:, 1, 2] - tri[:, 2, 2])
    )
    f = (
        (p[:, 0] - tri[:, 2, 0]) * (tri[:, 0, 0] - tri[:, 2, 0])
        + (p[:, 1] - tri[:, 2, 1]) * (tri[:, 0, 1] - tri[:, 2, 1])
        + (p[:, 2] - tri[:, 2, 2]) * (tri[:, 0, 2] - tri[:, 2, 2])
    )
    g = (
        (p[:, 0] - tri[:, 2, 0]) * (tri[:, 1, 0] - tri[:, 2, 0])
        + (p[:, 1] - tri[:, 2, 1]) * (tri[:, 1, 1] - tri[:, 2, 1])
        + (p[:, 2] - tri[:, 2, 2]) * (tri[:, 1, 2] - tri[:, 2, 2])
    )
    chi = (d * f - b * g) / (a * d - b * c)
    eta = (-c * f + a * g) / (a * d - b * c)
    lambda1 = chi
    lambda2 = eta
    lambda3 = 1 - chi - eta
    return np.vstack((lambda1, lambda2, lambda3)).T


def correct_triangle_one(indices, icosphere, warped_coords, r, trial=0):
    neighbours = icosphere["neighbours"][indices[r][trial]]
    triangles = []
    for tri in np.arange(len(neighbours)):
        triangle = [
            indices[r][trial],
            neighbours[tri],
            neighbours[(tri + 1) % len(neighbours)],
        ]
        triangles.append(triangle)
    triangles = np.array(triangles)
    stacked_centre = np.tile(icosphere["coords"][r], (len(neighbours), 1))
    n_lambdas = barycentric_coordinates_matrix(stacked_centre, warped_coords[triangles])

    new_tri = (np.abs(n_lambdas - 0.5) <= (0.5 + 2e-5)).all(axis=1)

    return new_tri, triangles, n_lambdas


def correct_triangles(icosphere, indices, redos, warped_coords, lambdas):
    """function to correct triangles where original barycentric coordinate is
    not inside triangle
    icosphere - icosphere surf"""
    for r in redos:
        new_tri = [0]
        trial = -1
        while sum(new_tri) == 0 and trial < 2:
            trial += 1
            new_tri, triangles, n_lambdas = correct_triangle_one(indices, icosphere, warped_coords, r, trial=trial)
        if np.sum(new_tri) > 0:
            if np.sum(new_tri) > 1:
                new_tri = np.where(new_tri)[0][0]

            indices[r] = triangles[new_tri]
            lambdas[r] = n_lambdas[new_tri]
    return indices, lambdas


def spinning_coords(coords):
    """rotate coordinates"""
    rotation = special_ortho_group.rvs(3)
    new_coords = coords @ rotation
    return new_coords
