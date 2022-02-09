#ico class
import os
import numpy as np
import nibabel as nb
import copy
import time
from scipy import sparse 
import meld_classifier.mesh_tools as mt


#loads in all icosphere
class IcoSpheres():
    """Class to define cohort-level parameters such as subject ids, mesh"""
    def __init__(self, icosphere_path='../data/icospheres/'):
        """icosphere class
        icospheres at each level are stored in self.icospheres[1:7]
        autoloads & calculates:
        'coords': spherical coordinates
        'faces': triangle faces
        'polar_coords': theta & phi spherical coords
        'edges': all edges
        'adj_mat': sparse adjacency matrix"""
        self.icosphere_path = icosphere_path
        self.icospheres={}
        self.load_all_levels()
        
    def load_all_levels(self):
        for level in np.arange(7)+1:
            self.load_icosphere(level=level)
            self.calculate_edges(level=level)
            self.calculate_neighbours(level=level)
            self.polar_coords(level=level)
        return
        
    def load_icosphere(self,level=7):
        surf_nb = nb.load(os.path.join(self.icosphere_path,f'ico{level}.surf.gii'))
        self.icospheres[level]={'coords':surf_nb.darrays[0].data,
              'faces':surf_nb.darrays[1].data}
        return 
    
    def calculate_edges(self,level=7):
        surf=self.icospheres[level]
        surf['edges'] = np.vstack(
                [surf['faces'][:, :2], surf['faces'][:, 1:3], surf['faces'][:, [2, 0]],
                 #add self edges
                np.vstack([np.arange(len(surf['coords'])),np.arange(len(surf['coords']))]).T]
            )
        surf['adj_mat'] = sparse.coo_matrix(
                (np.ones(len(surf['edges']), np.uint8), (surf['edges'][:, 0], surf['edges'][:, 1])),
                shape=(len(surf["coords"]), len(surf["coords"])),
            ).tocsr()
        return
    
    def calculate_neighbours(self,level=7):
        self.icospheres[level]['neighbours'] = np.array(mt.get_neighbours_from_tris(self.icospheres[level]['faces']),
                                                        dtype=object)
        return

    def polar_coords(self,level=7):
        self.icospheres[level]['polar_coords'] = mt.spherical_np(self.icospheres[level]['coords'])[:,1:]
        return