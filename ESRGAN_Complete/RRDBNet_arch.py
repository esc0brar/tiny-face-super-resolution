# Standard library
import functools

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODEL DEFINITION
# =============================================================================


def make_layer(block, n_layers):
    """Stack ``n_layers`` copies of ``block`` into an ``nn.Sequential``.

    Args:
        block (Callable[[], nn.Module]): Factory that returns a new block.
        n_layers (int): Number of blocks to stack.

    Returns:
        nn.Sequential: The stacked blocks.
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block with 5 convolutions and dense connections."""

    def __init__(self, nf=64, gc=32, bias=True):
        """Build the 5-conv dense block.

        Args:
            nf (int): Number of input/output feature channels.
            gc (int): Growth channel (intermediate channel count).
            bias (bool): Whether convolutions use a bias term.
        """
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        """Apply the dense block with a residual scaling of 0.2.

        Args:
            x (torch.Tensor): Input feature map (N, nf, H, W).

        Returns:
            torch.Tensor: Output feature map of the same shape as ``x``.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        """Build three stacked dense blocks.

        Args:
            nf (int): Number of feature channels.
            gc (int): Growth channel inside each dense block.
        """
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        """Apply the three dense blocks with a residual scaling of 0.2.

        Args:
            x (torch.Tensor): Input feature map (N, nf, H, W).

        Returns:
            torch.Tensor: Output feature map of the same shape as ``x``.
        """
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator with an RRDB trunk and 4x nearest-neighbor upsampling."""

    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        """Build the network.

        Args:
            in_nc (int): Input channel count.
            out_nc (int): Output channel count.
            nf (int): Number of feature channels.
            nb (int): Number of RRDB blocks in the trunk.
            gc (int): Growth channel inside each dense block.
        """
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Run a forward pass producing a 4x super-resolved image.

        Args:
            x (torch.Tensor): LR image batch of shape (N, in_nc, H, W).

        Returns:
            torch.Tensor: SR image batch of shape (N, out_nc, 4H, 4W).
        """
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class RRDBNet_TinyFaces(RRDBNet):
    """RRDBNet variant tuned for tiny-face super-resolution."""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=30, gc=32):
        """Build the network with deeper RRDB trunk defaults.

        Args:
            in_nc (int): Input channel count.
            out_nc (int): Output channel count.
            nf (int): Number of feature channels.
            nb (int): Number of RRDB blocks in the trunk.
            gc (int): Growth channel inside each dense block.
        """
        super().__init__(
            in_nc=3, out_nc=3,
            nf=64,  # Increase to 128 if VRAM allows
            nb=30,   # More blocks for finer details
            gc=32
        )
    def forward(self, x):
        """Apply a shallow + deep residual pass without explicit upsampling.

        Args:
            x (torch.Tensor): Input image batch (N, in_nc, H, W).

        Returns:
            torch.Tensor: Output image batch (N, out_nc, H, W).
        """
        # Add shallow feature emphasis
        shallow_feat = self.conv_first(x)
        deep_feat = self.RRDB_trunk(shallow_feat)
        return self.conv_last(shallow_feat + deep_feat)  # Residual skip