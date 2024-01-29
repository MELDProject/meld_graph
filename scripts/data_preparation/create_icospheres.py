## Script used to create the different icosphere used at different level of the unet

from meld_classifier.meld_cohort import MeldCohort, MeldSubject
import os
import numpy as np
import nibabel as nb
import copy
import time


def calc_n_verts_and_tris_in_down(n_tris):
    """calculate number of vertices and triangles in downsampled mesh"""
    down_n_triangles = n_tris // 4
    down_n_verts = 2 + down_n_triangles // 2
    return down_n_verts, down_n_triangles


def save_gifti(filename, surf):
    """full path to filename
    surf - 'coords' and 'faces'"""
    pcoord = nb.gifti.GiftiCoordSystem(1, 1)
    pmeta = {
        "description": "icosphere",  # brief info here
        "GeometricType": "Sphere",  # an actual surface; could be 'Inflated', 'Hull', etc
        "AnatomicalStructurePrimary": "CortexLeft",  # the specific structure represented
        "AnatomicalStructureSecondary": "GrayWhite",  # if the above field is not specific enough
    }
    parray = nb.gifti.GiftiDataArray(
        surf["coords"],
        intent="NIFTI_INTENT_POINTSET",  # represents a set of points
        coordsys=pcoord,  # see above
        datatype="NIFTI_TYPE_FLOAT32",  # float type data
        meta=nb.gifti.GiftiMetaData.from_dict(pmeta),  # again, see above.
    )
    tmeta = {
        "TopologicalType": "Closed",  # a closed surface, could be 'Open', see spec
        "Description": "anything",  # brief info here
    }
    tcoord = nb.gifti.GiftiCoordSystem(0, 0)
    tarray = nb.gifti.GiftiDataArray(
        surf["faces"],
        intent="NIFTI_INTENT_TRIANGLE",  # triangle surface elements
        coordsys=tcoord,  # see above
        datatype="NIFTI_TYPE_INT32",  # integer indices
        meta=nb.gifti.GiftiMetaData.from_dict(tmeta),  # see above
    )
    gii = nb.gifti.GiftiImage(darrays=[parray, tarray])
    nb.save(gii, filename)


def downsample_mesh(surf):
    n_verts, n_tris = calc_n_verts_and_tris_in_down(len(surf["faces"]))
    new_surf = {}
    new_surf["coords"] = surf["coords"][:n_verts]
    # figure out completely removed triangles
    triangles_gone = (surf["faces"] >= n_verts).all(axis=1)
    #
    copy_faces = copy.deepcopy(surf["faces"])[~triangles_gone]
    new_surf["faces"] = np.zeros((n_tris, 3), dtype=int)
    t2 = time.time()
    for ti, t0 in enumerate(surf["faces"][triangles_gone]):
        if ti % 1000 == 0:
            t1 = time.time()
            print(ti, t1 - t2)
            t2 = time.time()

        bool_vec = (
            np.logical_or(
                copy_faces == t0[2],
                np.logical_or(copy_faces == t0[0], copy_faces == t0[1]),
            ).sum(axis=1)
            == 2
        )
        tris = copy_faces[bool_vec]
        new_tri = tris.T[(tris < n_verts).T]
        new_surf["faces"][ti] = new_tri
        # copy_faces=copy_faces[~bool_vec]

    return new_surf


if __name__ == "__main__":
    data_dir = "../data/icospheres"
    c = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix.hdf5", dataset=None)
    sphere_surf = {"coords": c.coords, "faces": c.surf["faces"]}
    save_gifti(os.path.join(data_dir, "ico7.surf.gii"), sphere_surf)
    downsampled_surf = downsample_mesh(sphere_surf)
    save_gifti(os.path.join(data_dir, "ico6.surf.gii"), downsampled_surf)
    downsampled_surf = downsample_mesh(downsampled_surf)
    save_gifti(os.path.join(data_dir, "ico5.surf.gii"), downsampled_surf)
    downsampled_surf = downsample_mesh(downsampled_surf)
    save_gifti(os.path.join(data_dir, "ico4.surf.gii"), downsampled_surf)
    downsampled_surf = downsample_mesh(downsampled_surf)
    save_gifti(os.path.join(data_dir, "ico3.surf.gii"), downsampled_surf)
    downsampled_surf = downsample_mesh(downsampled_surf)
    save_gifti(os.path.join(data_dir, "ico2.surf.gii"), downsampled_surf)
    downsampled_surf = downsample_mesh(downsampled_surf)
    save_gifti(os.path.join(data_dir, "ico1.surf.gii"), downsampled_surf)
