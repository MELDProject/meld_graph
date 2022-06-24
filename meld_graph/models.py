import torch_geometric
import torch.nn as nn

from meld_graph.icospheres import IcoSpheres
import torch
from meld_graph.spiralconv import SpiralConv

class GMMConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size, edges, edge_vectors):
        super(GMMConv, self).__init__()
        self.layer = torch_geometric.nn.GMMConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)
        self.edges = edges
        self.edge_vectors = edge_vectors

    def forward(self, x, device):
        return self.layer(x, self.edges, self.edge_vectors)
        
# define model
class MoNet(nn.Module):
    def __init__(self, num_features, layer_sizes = [], dim=2, kernel_size=3, icosphere_params={}, 
                conv_type='GMMConv', spiral_len=10,
                activation_fn='relu', **kwargs):
        """
        Model with only conv layers.

        dim: dim for GMMConv, dimension of coord representation - 2 or 3
        kernel_size: number of kernels (default 3)
        layer_sizes: (list) output size of each conv layer. a final linear layer for going to 2 (binary classification) is added
        num_features: number of input features (input size)
        icosphere_params: params passes to IcoShperes for edges, coords, and neighbours
        conv_type: "GMMConv" or "SpiralConv"
        spiral_len: number of neighbors included in each convolution (for SpiralConv) TODO implement dilation as well
       

        Model outputs log softmax scores. Need to call torch.exp to get probabilities
        """
        super(MoNet, self).__init__()
        # set class parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.conv_type = conv_type
        layer_sizes.insert(0, num_features)
        self.layer_sizes = layer_sizes
        # activation function to be used throughout
        if activation_fn == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_fn == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise NotImplementedErrror('activation_fn: '+activation_fn)

        # TODO when changing aggregation of hemis, might need to use different graph here 
        # TODO for different coord systems, pass arguments to IcoSpheres here
        # TODO ideally passed as params to IcoSpheres that then returns the correct graphs at the right levels
        self.icospheres = IcoSpheres(**icosphere_params)  # pseudo

        # TODO to device call necessary here because icoshperes need to be on GPU during conv layer init
        self.icospheres.to(self.device)

        #store n_vertices for batch rearrangement
        self.n_vertices = len(self.icospheres.icospheres[7]['coords'])
        # set up conv layers + final fcl
        conv_layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            print('conv', in_size, out_size)
            if self.conv_type == 'GMMConv':
                edges = self.icospheres.get_edges(level=7)
                edge_vectors = self.icospheres.get_edge_vectors(level=7)
                cl = GMMConv(in_size, out_size, dim=dim, kernel_size=kernel_size, edges=edges, edge_vectors=edge_vectors)
            elif self.conv_type == 'SpiralConv':
                indices = self.icospheres.get_spirals(level=7)
                # TODO several spiral_len? one per block? 
                # TODO implement dilations
                indices = indices[:,:spiral_len]
                cl = SpiralConv(in_size, out_size, indices=indices)
            else:
                raise NotImplementedError()

            conv_layers.append(cl)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc = nn.Linear(self.layer_sizes[-1], 2)
        

    def to(self, device, **kwargs):
        super(MoNet, self).to(device, **kwargs)
        #self.icospheres.to(device)
        self.device = device

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()
    
    def forward(self, data):
        batch_x = data
        #reshape input to batch,n_vertices
        original_shape = batch_x.shape
        batch_x = batch_x.view((batch_x.shape[0]//self.n_vertices, self.n_vertices,self.num_features))

        outputs = []
        for x in batch_x:
            for cl in self.conv_layers:
                x = cl(x, device=self.device)
                x = self.activation_function(x)
            # add final linear layer
            x = self.fc(x)
            x = nn.LogSoftmax(dim=1)(x)
            outputs.append(x)
        batch_x = torch.stack(outputs)
        #reshape output to batch, n_vertices
        x = batch_x.view((original_shape[0],-1))
        return [x]  # return list, as other models return multiple outputs


class MoNetUnet(nn.Module):
    def __init__(self, num_features, layer_sizes, dim=2, kernel_size=3, 
                 
                 icosphere_params={}, conv_type='GMMConv', spiral_len=10,
                 deep_supervision=[],
                 activation_fn='relu'):
        """
        Unet model
        dim: dim for GMMConv, dimension of coord representation - 2 or 3 (for GMMConv)
        kernel_size: number of kernels (default 3) (for GMMConv)
        layer_sizes: (list of lists) per block, output size of each conv layer. This structure is mirrored in the decoder. a final linear layer for going to 2 (binary classification) is added
        num_features: number of input features (input size)
        icosphere_params: params passes to IcoShperes for edges, coords, and neighbours
        conv_type: "GMMConv" or "SpiralConv"
        spiral_len: number of neighbors included in each convolution (for SpiralConv) TODO implement dilation as well
        deep_supervision: list of levels at which deep supervision should be added, adds linear "squeeze" layer to the end of the block, and outputs these levels.
        Model outputs log softmax scores. Need to call torch.exp to get probabilities
        """
        super(MoNetUnet, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.conv_type = conv_type
        self.deep_supervision = sorted(deep_supervision)
        if activation_fn == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_fn == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise NotImplementedErrror('activation_fn: '+activation_fn)
        # TODO when changing aggregation of hemis, might need to use different graph here 
        # TODO for different coord systems, pass arguments to IcoSpheres here
        # TODO ideally passed as params to IcoSpheres that then returns the correct graphs at the right levels
        self.icospheres = IcoSpheres(**icosphere_params)  # pseudo

        # TODO to device call necessary here because icoshperes need to be on GPU during conv layer init
        self.icospheres.to(self.device)
        
        # set up conv + pooling layers - encoder
        encoder_conv_layers = []
        pool_layers = []
        num_features_on_skip = []

        num_blocks = len(layer_sizes)
        assert(num_blocks <= 7)  # cannot pool more levels than icospheres
        in_size = self.num_features
        level = 7
        #store n_vertices for batch rearrangement
        self.n_vertices = len(self.icospheres.icospheres[level]['coords'])
        for i in range(num_blocks):
            block = []
            print('encoder block', i, 'at level', level)
            for j,out_size in enumerate(layer_sizes[i]):
                # create conv layers
                print('conv', in_size, out_size)
                if self.conv_type == 'GMMConv':
                    edges = self.icospheres.get_edges(level=level)
                    edge_vectors = self.icospheres.get_edge_vectors(level=level)
                    cl = GMMConv(in_size, out_size, dim=dim, kernel_size=kernel_size, edges=edges, edge_vectors=edge_vectors)
                elif self.conv_type == 'SpiralConv':
                    indices = self.icospheres.get_spirals(level=level)
                    # TODO several spiral_len? one per block? 
                    # TODO implement dilations
                    indices = indices[:,:spiral_len]
                    cl = SpiralConv(in_size, out_size, indices=indices)
                else:
                    raise NotImplementedError()
                block.append(cl)
                in_size = out_size
            print('skip features for block', i, out_size)
            num_features_on_skip.append(out_size)
            encoder_conv_layers.append(nn.ModuleList(block))
            # only pool if not in last block
            if i < num_blocks-1:
                print('pool for block', i)
                level -= 1
                neigh_indices = self.icospheres.get_neighbours(level=level)
                pool_layers.append(HexPool(neigh_indices=neigh_indices))
        self.encoder_conv_layers = nn.ModuleList(encoder_conv_layers)
        self.pool_layers = nn.ModuleList(pool_layers)

        # - decoder going from lowest level up, but don't need to do the bottom block, is already in encoder
        # start with uppooling
        decoder_conv_layers = []
        unpool_layers = []
        deep_supervision_fcs = {}
        for i in range(num_blocks-1)[::-1]:
            # check if want deep supervision for this level
            if level in deep_supervision:
                deep_supervision_fcs[str(level)] = nn.Linear(in_size, 2)
            level += 1
            print('decoder block', i, 'at level', level)
            print('adding unpool to level', level)
            num = len(self.icospheres.get_neighbours(level=level))
            upsample = self.icospheres.get_upsample(target_level=level)
            unpool_layers.append(HexUnpool(upsample_indices=upsample, target_size=num))
            block = []
            for j,out_size in enumerate(layer_sizes[i][::-1]):
                if j == 0:
                    in_size = in_size+num_features_on_skip[i]
                    print('skip features', num_features_on_skip[i])
                
                print('adding conv ', in_size, out_size)
                if self.conv_type == 'GMMConv':
                    edges = self.icospheres.get_edges(level=level)
                    edge_vectors = self.icospheres.get_edge_vectors(level=level)
                    cl = GMMConv(in_size, out_size, dim=dim, kernel_size=kernel_size, edges=edges, edge_vectors=edge_vectors)
                elif self.conv_type == 'SpiralConv':
                    indices = self.icospheres.get_spirals(level=level)
                    indices = indices[:,:spiral_len]
                    cl = SpiralConv(in_size, out_size, indices=indices)
                else:
                    raise NotImplementedError()
                block.append(cl)
                in_size = out_size
            decoder_conv_layers.append(nn.ModuleList(block))
        self.decoder_conv_layers = nn.ModuleList(decoder_conv_layers)
        self.unpool_layers = nn.ModuleList(unpool_layers)
        self.deep_supervision_fcs = nn.ModuleDict(deep_supervision_fcs)
        self.fc = nn.Linear(in_size, 2)

    def to(self, device, **kwargs):
        super(MoNetUnet, self).to(device, **kwargs)
        #self.icospheres.to(device)
        self.device = device
    
    def forward(self, data):
        batch_x = data
        #reshape input to batch,n_vertices
        original_shape = batch_x.shape
        
        batch_x = batch_x.view((batch_x.shape[0]//self.n_vertices, self.n_vertices,self.num_features))
        skip_connections = []
        outputs=[]
        deep_output = {level: [] for level in self.deep_supervision}
        for x in batch_x:
            level = 7
            for i, block in enumerate(self.encoder_conv_layers):
                for cl in block:
                    x = cl(x, device=self.device)
                    x = self.activation_function(x)
                skip_connections.append(x)
                # apply pool except on last block
                if i < len(self.encoder_conv_layers)-1:
                    level -= 1
                    x = self.pool_layers[i](x)

            for i, block in enumerate(self.decoder_conv_layers):
                # check if want deep supervision for this level
                if level in self.deep_supervision:
                    x_out = self.deep_supervision_fcs[str(level)](x)
                    x_out = nn.LogSoftmax(dim=1)(x_out)
                    deep_output[level].append(x_out)
                skip_i = len(self.decoder_conv_layers)-1-i
                level += 1
                x = self.unpool_layers[i](x, device=self.device)
                x = torch.cat([x, skip_connections[skip_i]], dim=1)
                for cl in block:
                    x = cl(x, device=self.device)
                    x = self.activation_function(x)
            # add final linear layer
            x = self.fc(x)
            x = nn.LogSoftmax(dim=1)(x)
            outputs.append(x)
        batch_x = torch.stack(outputs)
        #reshape output to batch, n_vertices
        x = batch_x.view((original_shape[0],-1))
        #print([deep_output[level][0].shape for level in self.deep_supervision])
        deep_output = [torch.stack(deep_output[level]) for level in self.deep_supervision]
        #print([o.shape for o in deep_output])
        deep_output = [out.view(-1, 2) for out in deep_output]
        return [x] + deep_output

class HexPool(nn.Module):
    def __init__(self, neigh_indices):
        super(HexPool, self).__init__()
        self.neigh_indices = neigh_indices
        
    def forward(self, x):
        x = x[:len(self.neigh_indices)][self.neigh_indices]
        x = torch.max(x, dim=1)[0]
        return x

class HexUnpool(nn.Module):
    def __init__(self, upsample_indices, target_size):
        super(HexUnpool, self).__init__()
        self.upsample_indices = upsample_indices
        self.target_size = target_size
        
    def forward(self, x, device):
        limit = int(x.shape[0])
        new_x = torch.zeros(self.target_size,x.shape[1]).to(device)
        new_x[:limit] = x
        new_x[limit:] = torch.mean(x[self.upsample_indices],dim=1)
        return new_x

