import os
import numpy as np
import nibabel as nb
import copy
import time
from scipy import sparse 
import meld_classifier.mesh_tools as mt
import torch
from math import pi 




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
            self.load_icosphere(level = level)
            self.calculate_edges(level = level)
            self.calculate_neighbours(level = level)
            self.spherical_coords(level = level)
            self.pseudo_edge_attrs(level = level)
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
        self.icospheres[level]['neighbours'] = np.array(self.get_neighbours_from_tris(self.icospheres[level]['faces']),
                                                        dtype=object)
        
        return

    def spherical_coords(self,level=7):
        self.icospheres[level]['spherical_coords'] = mt.spherical_np(self.icospheres[level]['coords'])[:,1:]
        self.icospheres[level]['spherical_coords'][:,0] = self.icospheres[level]['spherical_coords'][:,0] - pi/2
        return
    
    def pseudo_edge_attrs(self,level=7):
        """pseudo edge attributes, difference between latitude and longitude"""
        col = self.icospheres[level]['edges'][:,0]
        row = self.icospheres[level]['edges'][:,1]
        pos = self.icospheres[level]['spherical_coords']
        pseudo = pos[col] - pos[row]
        alpha = pseudo[:,1]
        
        tmp = (alpha == 0).nonzero()[0]
        alpha[tmp] = 1e-15
        tmp = (alpha < 0).nonzero()[0]
        alpha[tmp] = np.pi + alpha[tmp]

        alpha = 2*np.pi + alpha
        alpha = np.remainder(alpha, 2*np.pi)
        pseudo[:,1]=alpha
        
        self.icospheres[level]['pseudo_edge_attr'] = torch.from_numpy(pseudo)
        self.icospheres[level]['edges'] = torch.from_numpy(self.icospheres[level]['edges'])
        return
    
   
        
    #helper functions
    def get_edges(self,level=7):
        
        return self.icospheres[level]['edges']
    
    def get_edge_vectors(self,level=7,dist_dtype ='pseudo'):
        if dist_dtype == 'pseudo':
            return self.icospheres[level]['pseudo_edge_attr']
        elif dist_dtype == 'exact':
            return self.icospheres[level]['exact_edge_attr']

    def get_neighbours_from_tris(self,tris):
        """Get surface neighbours from tris
        Input: tris
        Returns Nested list. Each list corresponds
        to the ordered neighbours for the given vertex"""
        n_vert = np.max(tris) + 1
        neighbours = [[] for i in range(n_vert)]
        for tri in tris:
            neighbours[tri[0]].extend([tri[1], tri[2]])
            neighbours[tri[2]].extend([tri[0], tri[1]])
            neighbours[tri[1]].extend([tri[2], tri[0]])
        # Get unique neighbours
        for k in range(len(neighbours)):
            neighbours[k] = [k]+self.f7(neighbours[k])
        return neighbours


    def f7(self,seq):
        """returns uniques but in order of original appearance.
        Used to retain neighbour triangle relationship"""
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]