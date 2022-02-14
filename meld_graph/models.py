from torch_geometric.nn import GMMConv
import torch.nn as nn

from meld_graph.icospheres import IcoSpheres
import torch

# define model
class MoNet(nn.Module):
    def __init__(self, num_features, layer_sizes, dim=2, kernel_size=3, icosphere_params={}):
        """
        Model with only conv layers.

        dim: dim for GMMConv, dimension of coord representation - 2 or 3
        kernel_size: number of kernels (default 3)
        layer_sizes: (list) output size of each conv layer. a final linear layer for going to 2 (binary classification) is added
        num_features: number of input features (input size)
        icosphere_params: params passes to IcoShperes for edges, coords, and neighbours
        
        Model outputs log softmax scores. Need to call torch.exp to get probabilities
        """
        super(MoNet, self).__init__()
        self.num_features = num_features
        assert len(layer_sizes) >= 1
        layer_sizes.insert(0, num_features)
        self.layer_sizes = layer_sizes
        conv_layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            conv_layers.append(GMMConv(in_size, out_size, dim=dim, kernel_size=kernel_size))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc = nn.Linear(self.layer_sizes[-1], 2)
        self.activation_function = nn.ReLU()
        # TODO when changing aggregation of hemis, might need to use different graph here 
        # TODO for different coord systems, pass arguments to IcoSpheres here
        # TODO ideally passed as params to IcoSpheres that then returns the correct graphs at the right levels
        self.icospheres = IcoSpheres(**icosphere_params)  # pseudo
        # initialise
        #self.reset_parameters()

    def to(self, device, **kwargs):
        super(MoNet, self).to(device, **kwargs)
        self.icospheres.to(device)

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()
    
    def forward(self, data):
        x = data
        for cl in self.conv_layers:
            x = cl(x, self.icospheres.get_edges(level=7), self.icospheres.get_edge_vectors(level=7))
            x = self.activation_function(x)
        # add final linear layer
        x = self.fc(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x

class MoNetUnet(nn.Module):
    def __init__(self, num_features, layer_sizes, dim=2, kernel_size=3, icosphere_params={}):
        """
        Unet model
        dim: dim for GMMConv, dimension of coord representation - 2 or 3
        kernel_size: number of kernels (default 3)
        layer_sizes: (list of lists) per block, output size of each conv layer. This structure is mirrored in the decoder. a final linear layer for going to 2 (binary classification) is added
        num_features: number of input features (input size)
        icosphere_params: params passes to IcoShperes for edges, coords, and neighbours
        
        Model outputs log softmax scores. Need to call torch.exp to get probabilities
        """
        super(MoNetUnet, self).__init__()
        self.device = None
        self.num_features = num_features

        self.activation_function = nn.ReLU()
        # TODO when changing aggregation of hemis, might need to use different graph here 
        # TODO for different coord systems, pass arguments to IcoSpheres here
        # TODO ideally passed as params to IcoSpheres that then returns the correct graphs at the right levels
        self.icospheres = IcoSpheres(**icosphere_params)  # pseudo
        
        # set up conv + pooling layers - encoder
        encoder_conv_layers = []
        pool_layers = []
        num_features_on_skip = []

        num_blocks = len(layer_sizes)
        assert(num_blocks <= 7)  # cannot pool more levels than icospheres
        in_size = self.num_features
        for i in range(num_blocks):
            block = []
            print('encoder block', i)
            for j,out_size in enumerate(layer_sizes[i]):
                # create conv layers
                print('conv', in_size, out_size)
                block.append(GMMConv(in_size, out_size, dim=dim, kernel_size=kernel_size))
                in_size = out_size
            print('skip features for block', i, out_size)
            num_features_on_skip.append(out_size)
            encoder_conv_layers.append(nn.ModuleList(block))
            # only pool if not in last block
            if i < num_blocks-1:
                print('pool for block', i)
                pool_layers.append(HexPool())
        self.encoder_conv_layers = nn.ModuleList(encoder_conv_layers)
        self.pool_layers = nn.ModuleList(pool_layers)

        # - decoder going from lowest level up, but don't need to do the bottom block, is already in encoder
        # start with uppooling
        decoder_conv_layers = []
        unpool_layers = []
        for i in range(num_blocks-1)[::-1]:
            print('decoder block', i)
            print('adding unpool')
            unpool_layers.append(HexUnpool())
            block = []
            for j,out_size in enumerate(layer_sizes[i][::-1]):
                if j == 0:
                    in_size = in_size+num_features_on_skip[i]
                    print('skip features', num_features_on_skip[i])
                
                print('adding conv ', in_size, out_size)
                block.append(GMMConv(in_size, out_size, dim=dim, kernel_size=kernel_size))
                in_size = out_size
            decoder_conv_layers.append(nn.ModuleList(block))
        self.decoder_conv_layers = nn.ModuleList(decoder_conv_layers)
        self.unpool_layers = nn.ModuleList(unpool_layers)

        self.fc = nn.Linear(in_size, 2)

    def to(self, device, **kwargs):
        super(MoNetUnet, self).to(device, **kwargs)
        self.icospheres.to(device)
        self.device = device
    
    def forward(self, data):
        x = data
        level = 7
        skip_connections = []
        for i, block in enumerate(self.encoder_conv_layers):
            #print('block', i)
            for cl in block:
                #print('conv at level', level)
                # apply cl
                x = cl(x, self.icospheres.get_edges(level=level), self.icospheres.get_edge_vectors(level=level))
                x = self.activation_function(x)
            skip_connections.append(x)
            # apply pool except on last block
            if i < len(self.encoder_conv_layers)-1:
                level -= 1
                #print('pool to level', level)
                neigh = self.icospheres.get_neighbours(level=level)
                x = self.pool_layers[i](x, neigh_indices=neigh)
                
        for i, block in enumerate(self.decoder_conv_layers):
            skip_i = len(self.decoder_conv_layers)-1-i
            #print('decoder block', i, 'skip_i', skip_i)
            level += 1
            #print('unpool to level', level)
            num = len(self.icospheres.get_neighbours(level=level))
            upsample = self.icospheres.get_upsample(target_level=level)
            x = self.unpool_layers[i](x, upsample_indices=upsample, target_size=num, device=self.device)

            x = torch.cat([x, skip_connections[skip_i]], dim=1)
            for cl in block:
                # apply conv layers
                #print('conv at level', level)
                x = cl(x, self.icospheres.get_edges(level=level), self.icospheres.get_edge_vectors(level=level))
                x = self.activation_function(x)

        # add final linear layer
        x = self.fc(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x

class HexPool(nn.Module):
        
    def forward(self, x, neigh_indices):
        x = x[:len(neigh_indices)][neigh_indices]
        x = torch.max(x, dim=1)[0]
        return x

class HexUnpool(nn.Module):
        
    def forward(self, x, upsample_indices, target_size, device):
        limit = int(x.shape[0])
        new_x = torch.zeros(target_size,x.shape[1]).to(device)
        new_x[:limit] = x
        new_x[limit:] = torch.mean(x[upsample_indices],dim=1)
        return new_x
