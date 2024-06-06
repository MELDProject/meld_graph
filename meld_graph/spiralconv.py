import torch
import torch.nn as nn
import logging
from torch_geometric.nn import InstanceNorm


class SpiralConv(nn.Module):
    """
    Spiral convolution, adapted from https://github.com/sw-gong/spiralnet_plus/blob/master/conv/spiralconv.py
    """

    def __init__(self, in_channels, out_channels, indices, norm=None):
        super(SpiralConv, self).__init__()
        self.log = logging.getLogger(__name__)
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        if norm is not None:
            if norm == "instance":
                self.log.debug("Spiral Conv: Using instance norm")
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
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x, device):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.contiguous().to(device).view(-1))
            x = x.view(n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def __repr__(self):
        return "{}({}, {}, seq_length={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.seq_length,
        )
