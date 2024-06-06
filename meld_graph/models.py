import torch_geometric
import torch.nn as nn
from copy import deepcopy

from meld_graph.icospheres import IcoSpheres
import torch
from meld_graph.spiralconv import SpiralConv
from torch_geometric.nn import InstanceNorm
import scipy.sparse as sp
import numpy as np


class GMMConv(nn.Module):
    """
    GMMConv implementation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dim,
        kernel_size,
        edges,
        edge_vectors,
        norm=None,
    ):
        super(GMMConv, self).__init__()
        self.layer = torch_geometric.nn.GMMConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)
        self.edges = edges
        self.edge_vectors = edge_vectors
        if norm is not None:
            if norm == "instance":
                self.norm = InstanceNorm(
                    in_channels=out_channels,
                    eps=1e-05,
                    momentum=0.1,
                    affine=False,
                    track_running_stats=False,
                )
            else:
                raise NotImplementedError(norm)
        else:
            self.norm = None

    def forward(self, x, device):
        x = self.layer(x, self.edges, self.edge_vectors)
        if self.norm is not None:
            x = self.norm(x)
        return x


class MoNet(nn.Module):
    """
    Graph CNN Model for lesion segmentation with only convolutional layers.

    Model outputs are a dictionary containing
        'log_softmax' (log softmax lesion segmentation output),
        'non_lesion_logits' (non lesion output for distance regression)

    NOTE this model does not support distance regression and hemisphere classification. Use MoNetUnet instead.
    NOTE the model ouputs log softmax scores. Need to call torch.exp to get probabilities.

    Args:
        num_features (int): number of input features
        layer_sizes (list): output size of each conv layer. a final linear layer for going to 2 (binary classification) is added.
        dim (int): dim for GMMConv, dimension of coord representation - 2 or 3 (for GMMConv).
        kernel_size (int): number of kernels (default 3) (for GMMConv).
        icosphere_params (dict): params passed to IcoShperes for edges, coords, and neighbours.
        conv_type (str): type of convolution. "GMMConv" or "SpiralConv".
        spiral_len (int): number of neighbors included in each convolution (for SpiralConv).
        activation_fn (str): activation function. "relu" or "leaky_relu".
    """

    def __init__(
        self,
        num_features,
        layer_sizes=[],
        dim=2,
        kernel_size=3,
        icosphere_params={},
        conv_type="GMMConv",
        spiral_len=7,
        activation_fn="relu",
        norm=None,
    ):
        super(MoNet, self).__init__()
        # set class parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_features = num_features
        self.conv_type = conv_type
        # ensure that we don't have side effects when modifying layer sizes
        layer_sizes = deepcopy(layer_sizes)
        layer_sizes.insert(0, num_features)
        self.layer_sizes = layer_sizes
        # activation function to be used throughout
        if activation_fn == "relu":
            self.activation_function = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation_function = nn.LeakyReLU()
        else:
            raise NotImplementedErrror("activation_fn: " + activation_fn)

        self.icospheres = IcoSpheres(**icosphere_params)
        # NOTE to device call necessary here because icoshperes need to be on GPU during conv layer init
        self.icospheres.to(self.device)

        # store n_vertices for batch rearrangement
        self.n_vertices = len(self.icospheres.icospheres[7]["coords"])
        # set up conv layers + final fcl
        conv_layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # print('conv', in_size, out_size)
            if self.conv_type == "GMMConv":
                edges = self.icospheres.get_edges(level=7)
                edge_vectors = self.icospheres.get_edge_vectors(level=7)
                cl = GMMConv(
                    in_size,
                    out_size,
                    dim=dim,
                    kernel_size=kernel_size,
                    edges=edges,
                    edge_vectors=edge_vectors,
                    norm=norm,
                )
            elif self.conv_type == "SpiralConv":
                indices = self.icospheres.get_spirals(level=7)
                indices = indices[:, :spiral_len]
                cl = SpiralConv(in_size, out_size, indices=indices, norm=norm)
            else:
                raise NotImplementedError()

            conv_layers.append(cl)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc = nn.Linear(self.layer_sizes[-1], 2)

    def to(self, device, **kwargs):
        super(MoNet, self).to(device, **kwargs)
        # self.icospheres.to(device)
        self.device = device

    def reset_parameters(self):
        for layer in self.conv_layers:
            layer.reset_parameters()

    def forward(self, data):
        batch_x = data
        # reshape input to batch,n_vertices
        original_shape = batch_x.shape
        batch_x = batch_x.view((batch_x.shape[0] // self.n_vertices, self.n_vertices, self.num_features))

        outputs = {"log_softmax": [], "non_lesion_logits": []}
        for x in batch_x:
            for cl in self.conv_layers:
                x = cl(x, device=self.device)
                x = self.activation_function(x)
            # add final linear layer
            x = self.fc(x)
            outputs["non_lesion_logits"].append(x[:, 0])
            x = nn.LogSoftmax(dim=1)(x)
            outputs["log_softmax"].append(x)

        # stack and reshape outputs to (batch * n_vertices, -1)
        for key, output in outputs.items():
            shape = (-1, 2)
            if "non_lesion_logits" in key:
                shape = (-1, 1)
            outputs[key] = torch.stack(output).view(shape)
        return outputs


class MoNetUnet(nn.Module):
    """
    Graph U-Net Model for lesion segmentation with additional distance regression and hemisphere classification.

    Model outputs are a dictionary containing
        'log_softmax' (log softmax lesion segmentation output),
        'non_lesion_logits' (non lesion output for distance regression)
        'hemi_log_softmax' (classification head output, if classification_head is True)
        'object_detection_linear' (object detection head output, if object_detection_head is True)
    For every deep supervision layer, the output are: 'ds{level}_log_softmax', 'ds{level}_non_lesion_logits'

    NOTE the model ouputs log softmax scores. Need to call torch.exp to get probabilities.

    Args:
        num_features (int): number of input features
        layer_sizes (list): list of lists, defining the number of filters in every conv layer in every block of the U-Net.
            Only the downsampling branch and the bottom block needs to be defined.
            Upsampling will mirror layer_sizes[:-1:-1].
            A final layer for log_softmax output and a classification head will be added.
        dim (int): dim for GMMConv, dimension of coord representation - 2 or 3 (for GMMConv).
        kernel_size (int): number of kernels (default 3) (for GMMConv).
        icosphere_params (dict): params passed to IcoShperes for edges, coords, and neighbours.
        conv_type (str): type of convolution. "GMMConv" or "SpiralConv".
        spiral_len (int): number of neighbors included in each convolution (for SpiralConv).
        deep_supervision (list): list of levels at which deep supervision should be added, adds linear "squeeze" layer to the end of the block, and outputs these levels.
        activation_fn (str): activation function. "relu" or "leaky_relu".
        norm (str): "instance" or None
        classification_head (bool): should a subject classification head be created from the lowest level. 
            Classification head contains linear "squeeze" layers over features and over all vertices, resulting in one output per hemisphere. 
        object_detection_head (bool): should an object detection head be created from the lowest level.
            Object detection head contains linear "squeeze" layers over features and over all vertices, resulting in 4 outputs per hemisphere.

    """
    def __init__(self, 
                 num_features, 
                 layer_sizes, 
                 dim=2, 
                 kernel_size=3, 
                 icosphere_params={}, 
                 conv_type='SpiralConv', 
                 spiral_len=7,
                 deep_supervision=[],
                 activation_fn='relu', 
                 norm=None,
                 classification_head=False,
                 distance_head=False,
                 object_detection_head=False
                 ):
        super(MoNetUnet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_features = num_features
        self.conv_type = conv_type
        self.deep_supervision = sorted(deep_supervision)
        self.classification_head = classification_head
        self.distance_head = distance_head
        self.object_detection_head = object_detection_head
        if activation_fn == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation_function = nn.LeakyReLU()
        else:
            raise NotImplementedErrror("activation_fn: " + activation_fn)

        self.icospheres = IcoSpheres(**icosphere_params)  # pseudo
        # NOTE to device call necessary here because icoshperes need to be on GPU during conv layer init
        self.icospheres.to(self.device)

        # set up conv + pooling layers - encoder
        encoder_conv_layers = []
        pool_layers = []
        num_features_on_skip = []

        num_blocks = len(layer_sizes)
        assert num_blocks <= 7  # cannot pool more levels than icospheres
        in_size = self.num_features
        level = 7
        # store n_vertices for batch rearrangement
        self.n_vertices = len(self.icospheres.icospheres[level]["coords"])
        for i in range(num_blocks):
            block = []
            for j, out_size in enumerate(layer_sizes[i]):
                # create conv layers
                if self.conv_type == "GMMConv":
                    edges = self.icospheres.get_edges(level=level)
                    edge_vectors = self.icospheres.get_edge_vectors(level=level)
                    cl = GMMConv(
                        in_size,
                        out_size,
                        dim=dim,
                        kernel_size=kernel_size,
                        edges=edges,
                        edge_vectors=edge_vectors,
                        norm=norm,
                    )
                elif self.conv_type == "SpiralConv":
                    indices = self.icospheres.get_spirals(level=level)
                    indices = indices[:, :spiral_len]
                    cl = SpiralConv(in_size, out_size, indices=indices, norm=norm)
                else:
                    raise NotImplementedError()
                block.append(cl)
                in_size = out_size
            num_features_on_skip.append(out_size)
            encoder_conv_layers.append(nn.ModuleList(block))
            # only pool if not in last block
            if i < num_blocks - 1:
                level -= 1
                neigh_indices = self.icospheres.get_downsample(target_level=level)
                pool_layers.append(HexPool(neigh_indices=neigh_indices))
        self.encoder_conv_layers = nn.ModuleList(encoder_conv_layers)
        self.pool_layers = nn.ModuleList(pool_layers)
        if self.classification_head:
            # go from all vertices at lowest level to 2 nodes. First aggregated over kernel size, then over vertices
            self.hemi_classification_head = nn.ModuleList([
                nn.Conv1d(in_size, 1, kernel_size=1),
                nn.Linear(len(self.icospheres.icospheres[level]['coords']), 2)
                ])
            
        if self.object_detection_head:
            #object detection head
            self.object_detection_head = nn.ModuleList([
                nn.Conv1d(in_size, 1, kernel_size=1),
                nn.Linear(len(self.icospheres.icospheres[level]['coords']), 4)
            ])
                

        # - decoder going from lowest level up, but don't need to do the bottom block, is already in encoder
        # start with uppooling
        decoder_conv_layers = []
        unpool_layers = []
        deep_supervision_fcs = {}
        if self.distance_head:
            distance_fcs = {}
        for i in range(num_blocks - 1)[::-1]:
            # check if want deep supervision for this level
            if level in deep_supervision:
                deep_supervision_fcs[str(level)] = nn.Linear(in_size, 2)
                if self.distance_head:
                    distance_fcs[str(level)] = nn.Linear(in_size, 1)
            level += 1
            num = len(self.icospheres.get_neighbours(level=level))
            upsample = self.icospheres.get_upsample(target_level=level)
            unpool_layers.append(HexUnpool(upsample_indices=upsample, target_size=num))
            block = []
            for j, out_size in enumerate(layer_sizes[i][::-1]):
                if j == 0:
                    in_size = in_size + num_features_on_skip[i]

                if self.conv_type == "GMMConv":
                    edges = self.icospheres.get_edges(level=level)
                    edge_vectors = self.icospheres.get_edge_vectors(level=level)
                    cl = GMMConv(
                        in_size,
                        out_size,
                        dim=dim,
                        kernel_size=kernel_size,
                        edges=edges,
                        edge_vectors=edge_vectors,
                    )
                elif self.conv_type == "SpiralConv":
                    indices = self.icospheres.get_spirals(level=level)
                    indices = indices[:, :spiral_len]
                    cl = SpiralConv(in_size, out_size, indices=indices)
                else:
                    raise NotImplementedError()
                block.append(cl)
                in_size = out_size
            decoder_conv_layers.append(nn.ModuleList(block))
        self.decoder_conv_layers = nn.ModuleList(decoder_conv_layers)
        self.unpool_layers = nn.ModuleList(unpool_layers)
        self.deep_supervision_fcs = nn.ModuleDict(deep_supervision_fcs)
        if self.distance_head:
            self.distance_fcs = nn.ModuleDict(distance_fcs)
            self.distance_fc = nn.Linear(in_size, 1)
        self.fc = nn.Linear(in_size, 2)

    def to(self, device, **kwargs):
        super(MoNetUnet, self).to(device, **kwargs)
        self.icospheres.to(device)
        self.device = device

    def forward(self, data):
        batch_x = data
        # reshape input to batch,n_vertices
        original_shape = batch_x.shape

        batch_x = batch_x.view((batch_x.shape[0] // self.n_vertices, self.n_vertices, self.num_features))
        skip_connections = []
        outputs = {"log_softmax": [], "non_lesion_logits": [], "log_sumexp": []}
        for level in self.deep_supervision:
            outputs[f"ds{level}_log_softmax"] = []
            outputs[f"ds{level}_non_lesion_logits"] = []
            outputs[f"ds{level}_log_sumexp"] = []
        if self.classification_head:
            outputs['hemi_log_softmax'] = []
        if self.object_detection_head:
            outputs['object_detection_linear'] = []
        for x in batch_x:
            level = 7
            for i, block in enumerate(self.encoder_conv_layers):
                for cl in block:
                    x = cl(x, device=self.device)
                    x = self.activation_function(x)
                skip_connections.append(x)
                # apply pool except on last block
                if i < len(self.encoder_conv_layers) - 1:
                    level -= 1
                    x = self.pool_layers[i](x)

            if self.classification_head:
                hemi_classification = self.activation_function(self.hemi_classification_head[0](x.unsqueeze(2)))
                hemi_classification = self.hemi_classification_head[1](hemi_classification.view(-1))
                hemi_classification = nn.LogSoftmax(dim=0)(hemi_classification)
                outputs["hemi_log_softmax"].append(hemi_classification)

            if self.object_detection_head:
                # Object detection head
                xyzr = self.object_detection_head[0](x.unsqueeze(2))
                xyzr = self.object_detection_head[1](xyzr.view(-1))
                xyz = xyzr[:3].contiguous()
                r = xyzr[ 3:].contiguous()
                xyz = xyz.view(xyz.size(0), -1)
                xyz_norm = nn.functional.normalize(xyz, p=2, dim=0)
                xyzr_norm = torch.cat([xyz_norm, r.view(r.size(0), -1)], dim=0)
                xyzr_norm = torch.squeeze(xyzr_norm)
                outputs['object_detection_linear'].append(xyzr_norm)

            for i, block in enumerate(self.decoder_conv_layers):
                # check if want deep supervision for this level
                if level in self.deep_supervision:
                    x_out = self.deep_supervision_fcs[str(level)](x)
                    if self.distance_head:
                        x_dist = self.distance_fcs[str(level)](x)
                        outputs[f"ds{level}_non_lesion_logits"].append(x_dist)
                    else:
                        outputs[f"ds{level}_non_lesion_logits"].append(x_out[:, 0])
                    x_out = nn.LogSoftmax(dim=1)(x_out)
                    outputs[f"ds{level}_log_softmax"].append(x_out)

                    x_out_logsumexp = torch.logsumexp(torch.exp(x_out[:, 1]), dim=0)
                    x_out_logsumexp = torch.stack((1 - x_out_logsumexp, x_out_logsumexp))
                    x_out_logsumexp = nn.LogSoftmax(dim=0)(x_out_logsumexp)
                    outputs[f"ds{level}_log_sumexp"].append(x_out_logsumexp)

                skip_i = len(self.decoder_conv_layers) - 1 - i
                level += 1
                x = self.unpool_layers[i](x, device=self.device)
                x = torch.cat([x, skip_connections[skip_i]], dim=1)
                for cl in block:
                    x = cl(x, device=self.device)
                    x = self.activation_function(x)

            # add distance head
            if self.distance_head:
                x_dist = self.distance_fc(x)
                outputs["non_lesion_logits"].append(x_dist)
            else:
                outputs["non_lesion_logits"].append(x[:, 0])
            # add final linear layer
            x = self.fc(x)
            x = nn.LogSoftmax(dim=1)(x)
            outputs["log_softmax"].append(x)

            x_logsumexp = torch.logsumexp(torch.exp(x[:, 1]), dim=0)
            x_logsumexp = torch.stack((1 - x_logsumexp, x_logsumexp))
            x_logsumexp = nn.LogSoftmax(dim=0)(x_logsumexp)
            outputs["log_sumexp"].append(x_logsumexp)

        # stack and reshape outputs to (batch * n_vertices, -1)
        #object detection is (batch * 4,-1)
        # in case of hemi classification will be (batch, -1)
        for key, output in outputs.items():
            shape = (-1, 2)
            if "non_lesion_logits" in key:
                shape = (-1, 1)
            if "object_detection_linear" in key:
                shape = (-1,4)
            outputs[key] = torch.stack(output).view(shape)
        return outputs


class HexPool(nn.Module):
    """
    Max pooling for Icospheres.
    """

    def __init__(self, neigh_indices):
        super(HexPool, self).__init__()
        self.neigh_indices = neigh_indices

    def forward(self, x, center_pool=False):
        """
        Forward pass

        Args:
            center_pool (bool): default is max pool, set center_pool to true to do center pool
        """
        if center_pool:
            x = x[: len(self.neigh_indices)]
        else:
            x = x[self.neigh_indices]
            x = torch.max(x, dim=1)[0]
        return x


class HexUnpool(nn.Module):
    """
    Mean unpooling for Icospheres.
    """

    def __init__(self, upsample_indices, target_size):
        super(HexUnpool, self).__init__()
        self.upsample_indices = upsample_indices
        self.target_size = target_size

    def forward(self, x, device):
        limit = int(x.shape[0])
        new_x = torch.zeros(self.target_size, x.shape[1]).to(device)
        new_x[:limit] = x
        new_x[limit:] = torch.mean(x[self.upsample_indices], dim=1)
        return new_x


class HexSmoothSparse(nn.Module):
    """
    Smoothing for Icospheres.
    """

    def __init__(self, neighbours):
        super(HexSmoothSparse, self).__init__()
        self.neighbours = neighbours
        self.sparse_weights = self._create_sparse_weights(neighbours)

    def _create_sparse_weights(self, neighbours):
        """
        Create sparse matrix with weights for the smoothing operation.
        """
        num_points = len(neighbours)
        num_neighbours = neighbours.shape[1]
        row_indices = np.repeat(np.arange(num_points), num_neighbours)
        col_indices = neighbours.reshape(-1)
        values = np.ones(row_indices.shape) / num_neighbours
        sparse_weights = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_points, num_points))
        return sparse_weights

    def forward(self, x):
        new_x = self.sparse_weights @ x
        return new_x


class PredictionForSaliency(nn.Module):
    """
    Adds layer to model to output the sum of all predicted lesional vertices for integrated gradients
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input, mask=None):
        prediction = torch.exp(self.model(input)['log_softmax'])
        if mask is None:
            prediction = torch.mean(prediction, axis=0)
        else:
            prediction = torch.mean(prediction[mask], axis=0)
        # return shape is (1,2)
        return prediction[None,:]