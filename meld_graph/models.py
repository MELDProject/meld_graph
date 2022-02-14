from torch_geometric.nn import GMMConv
import torch.nn as nn

from meld_graph.icospheres import IcoSpheres
import torch

# define model
class MoNet(nn.Module):
    def __init__(self, num_features, layer_sizes, dim=2, kernel_size=3, icosphere_params={}):
        """
        dim: dim for GMMConv, dimension of coord representation - 2 or 3
        kernel_size: number of kernels (default 3)
        layer_sizes: (list) output size of each conv layer. a final linear layer for going to 2 (binary classification) is added
        num_features: number of input features (input size)
        edge_index_fn: function that takes level as argument and returns (edge_index, edge_attrs)
        
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
        self.conv_layers = nn.Sequential(*conv_layers)
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

