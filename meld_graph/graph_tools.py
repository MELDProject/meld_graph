import numpy as np
import torch
import potpourri3d as pp3d
from meld_graph.models import HexUnpool, HexPool, HexSmooth

class GraphTools:
    def __init__(self, icospheres):
        """
        Use graph tools
        """
        self.icospheres = icospheres
        
        #initialise distance solver
        self.setup_distance_solver()
    
    def setup_distance_solver(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pool7 = self.pool(level=6)
        self.pool6 = self.pool(level=5)
        self.unpool6 = self.unpool(level=6)
        self.unpool7 = self.unpool(level=7)
        self.smooth5 = self.smoother(level=5)
        self.solver = pp3d.MeshHeatMethodDistanceSolver(self.icospheres.icospheres[5]['coords'],
                    self.icospheres.icospheres[5]['faces'])
        self.smoother = self.smoother(level=7)

    def pool(self,level=7):
        neigh_indices = self.icospheres.get_downsample(target_level=level)
        pooling = HexPool(neigh_indices=neigh_indices)
        return pooling

    def unpool(self,level=7):
        num = len(self.icospheres.get_neighbours(level=level))
        upsample = self.icospheres.get_upsample(target_level=level)
        unpooling = HexUnpool(upsample_indices=upsample, target_size=num)
        return unpooling
    
    def smoother(self,level=7):
        neighbours = self.icospheres.get_neighbours(level=level)
        pooling = HexSmooth(neighbours=neighbours)
        return pooling

    def smoothing(self, data, iteration=1):
        data = torch.from_numpy(data.astype(float)).to(self.device)
        for i in range(0, iteration):
            data = self.smoother(data)
        data = data.detach().cpu().numpy().ravel()
        return data
    
    def fast_geodesics(self,lesion):
        """calculate geodesic distances on downsampled mesh then upsample
        currently calculating on level 5, with two upsample steps"""

        #downsample lesion
        #if no lesion, no distance
        if lesion.sum()==0:
            n_vert = len(self.icospheres.icospheres[7]['coords'])
            return np.ones(n_vert)*200
        n_vert = len(self.icospheres.icospheres[5]['coords'])

        indices = np.arange(n_vert,dtype=int)
        downsampled1 = self.pool7(torch.from_numpy(lesion.reshape(-1,1)))
        lesion_small = self.pool6(downsampled1).detach().cpu().numpy().ravel()
        
        #find boundaries of lesions
        new_lesion = self.smooth5(torch.from_numpy(lesion_small.astype(float)).to(self.device))
        new_lesion = new_lesion.detach().cpu().numpy().ravel()
        lesion_boundary_vertices = indices[(lesion_small - new_lesion)>0]
        boundary_distance = self.solver.compute_distance_multisource(lesion_boundary_vertices)

        # upsample distance
        # boundary_distance[lesion_small == 1] = 0
        upsampled1 = self.unpool6(torch.from_numpy(boundary_distance.reshape(-1,1)),
        device=self.device)
        full_upsampled = self.unpool7(upsampled1, device = self.device)
        full_upsampled = full_upsampled.detach().cpu().numpy().ravel()
        
        #inverse values on the lesion
        full_upsampled[lesion>0]=-full_upsampled[lesion>0]
        
        return full_upsampled
    


