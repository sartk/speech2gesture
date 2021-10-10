import math
from collections import OrderedDict

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from resize_right import resize
import interp_methods

class AudioToPose(nn.Module):

    def __init__(self, pose_shape: Tuple[int, int], input_shape: Tuple[int, int], encoder_dim=2):
        super(AudioToPose, self).__init__()

        pose_dof, frames = pose_shape
        h, w = input_shape # h is time, w is features

        audio_encoder_mults = [1, 2, 1, 2, 1, 2, 1, 1]
        channel_mults = [1 if encoder_dim == 1 else 64, 2, 1, 2, 1, 1, 1, 1]

        self.encoder_dim = encoder_dim
        self.audio_encoder_down_factors, self.channel_factors = [1], [None, [64], [1]][encoder_dim]
        for m, n in zip(audio_encoder_mults, channel_mults):
            self.audio_encoder_down_factors.append(m * self.audio_encoder_down_factors[-1])
            self.channel_factors.append(n * self.channel_factors[-1])

        if encoder_dim == 1:
            sizes = [(cdiv(h, factor),) for factor in self.audio_encoder_down_factors]
        else:
            sizes = [(cdiv(h, factor), cdiv(w, factor)) for factor in self.audio_encoder_down_factors]

        channels = [factor for factor in self.channel_factors]

        self.audio_encoder = nn.ModuleList(
            [[None, ConvNormRelu1d, ConvNormRelu2d][encoder_dim](in_channels=channels[i], out_channels=channels[i + 1], leaky=True,
                            downsample=(audio_encoder_mults[i] > 1), input_shape=sizes[i],
                            output_shape=sizes[i + 1]) for i in range(len(sizes) - 1)])

        self.set_resize(frames)
        self.unet_encoder_labels = [5, 6, 7, 8, 9, 10]

        self.unet_encoder = nn.ModuleList([
            nn.Sequential(OrderedDict(
                ('conv1', ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(frames,))),
                ('conv2', ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(frames,)))
            )),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                           input_shape=(frames,), output_shape=(cdiv(frames, 2),)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                             input_shape=(cdiv(frames, 2),), output_shape=(cdiv(frames, 4),)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                            input_shape=(cdiv(frames, 4),), output_shape=(cdiv(frames, 8),)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                           input_shape=(cdiv(frames, 8),), output_shape=(cdiv(frames, 16),)),
            nn.Sequential(
                OrderedDict('conv1', ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                                         input_shape=(cdiv(frames, 16),),
                                                         output_shape=(cdiv(frames, 32),))),
                OrderedDict('upsample1', nn.Upsample(size=(cdiv(frames, 16),))))
        ])
        self.unet_decoder_labels = [9, 8, 7, 6, 5]
        self.unet_decoder = nn.ModuleList([
            nn.Sequential(
                OrderedDict()
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(frames, 16),)),
                nn.Upsample(size=(cdiv(frames, 8),))),
            nn.Sequential(
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(frames, 8),)),
                nn.Upsample(size=(cdiv(frames, 4),))),
            nn.Sequential(
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(frames, 4),)),
                nn.Upsample(size=(cdiv(frames, 2),))),
            nn.Sequential(
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(frames, 2),)),
                nn.Upsample(size=(frames,))),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                                           input_shape=(frames,))
        ])
        self.logits = nn.ModuleList([
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False, input_shape=(frames,)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False, input_shape=(frames,)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False, input_shape=(frames,)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False, input_shape=(frames,)),
            nn.Conv1d(in_channels=256, out_channels=pose_dof, kernel_size=(1,), stride=(1,))
        ])

        def weight_init(m):
            if m == self.logits[-1]:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

        self.apply(weight_init)


    def set_input_shape(self, shape: Tuple[int, int]):
        h, w = shape
        if encoder_dim == 1:
            sizes = [(cdiv(h, factor),) for factor in self.audio_encoder_down_factors]
        else:
            sizes = [(cdiv(h, factor), cdiv(w, factor)) for factor in self.audio_encoder_down_factors]
        for i in range(len(sizes) - 1):
            conv_block.reset_shape(sizes[i], sizes[i + 1])

    def set_output_shape(self, pose_dof, frames):


    def set_resize(self, frames):
        if encoder_dim == 2:
            self.resize = lambda t: resize(t, out_shape=(frames, 1), interp_method=interp_methods.linear).squeeze(-1)
        elif encoder_dim == 1:
            self.resize = lambda t: resize(t, out_shape=(frames,), interp_method=interp_methods.linear)

    def forward(self, x: Tensor) -> Tensor:
        if self.encoder_dim == 2:
            x = x.unsqueeze(1)
        elif self.encoder_dim == 1:
            x = x.permute(0, 2, 1) # bring features to front and time to end
        for conv_block in self.audio_encoder:
            x = conv_block(x)
        x = self.resize(x)
        skip_connections = dict()
        for name, conv_block in zip(self.unet_encoder_labels, self.unet_encoder):
            x = conv_block(x)
            skip_connections[name] = x
        for name, conv_block in zip(self.unet_decoder_labels, self.unet_decoder):
            x += skip_connections[name]
            x = conv_block(x)
        skip_connections.clear()
        for conv_block in self.logits:
            x = conv_block(x)
        return x


class PoseDiscriminator(nn.Module):

    def __init__(self, pose_shape, ndf=64, n_downsampling=2):
        super(PoseDiscriminator, self).__init__()
        pose_dof, frames = pose_shape
        padding = PoseDiscriminator._compute_padding(frames, 4, 2, cdiv(frames, 2))
        self.padding1 = (padding // 2, padding - padding // 2)
        self.conv1 = nn.Conv1d(in_channels=pose_dof, out_channels=ndf, kernel_size=(4,), stride=(2,))
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv_blocks = nn.ModuleList()

        in_channels = ndf
        input_size = cdiv(frames, 2)
        for n in range(1, n_downsampling):
            nf_mult = min(2 ** n, 8)
            self.conv_blocks.append(
                ConvNormRelu1d(in_channels=in_channels, out_channels=ndf * nf_mult, downsample=True, leaky=True,
                               input_shape=(input_size,), output_shape=(cdiv(input_size, 2),)))
            in_channels = ndf * nf_mult
            input_size = cdiv(input_size, 2)

        nf_mult = min(2 ** n_downsampling, 8)
        self.conv_blocks.append(ConvNormRelu1d(in_channels=in_channels, out_channels=ndf * nf_mult, kernel=(4,),
                                         stride=(1,), leaky=True, input_shape=(input_size,)))
        in_channels = ndf * nf_mult
        self.classifier = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=(4,), stride=(1,),
                                    padding='same')

        def weight_init(m):
            if m in [self.classifier, self.conv1]:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

        self.apply(weight_init)

    @staticmethod
    def _compute_padding(in_size, kernel, stride, out_size):
        return kernel + (out_size - 1) * stride - in_size

    def forward(self, x):
        x = F.pad(x, self.padding1)
        x = self.conv1(x)
        x = self.relu1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.classifier(x)
        return x


class ConvNormRelu(nn.Module):

    def __init__(self, leaky=False, downsample=False, kernel=None, stride=None, padding=None, input_shape=None,
                 output_shape=None, dim=2,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.output_shape = output_shape
        self.input_shape = input_shape
        if kernel is None and stride is None:
            kernel, stride = (3, 1) if not downsample else (4, 2)
        for attr, var in zip(['kernel', 'stride'], [kernel, stride]):
            # print('setting attr: ', attr, var)
            self._set_multidimensional_attr(attr, var, dim)
            # print('set attr', attr, getattr(self, attr))
        if padding is None:
            assert input_shape is not None, 'Either padding or input shape must be provided to ConvNormRelu.'
            padding = self.compute_padding(input_shape)
        self._set_multidimensional_attr('padding', padding, 2 * dim)
        self.downsample = downsample
        self.leaky = leaky
        if self.leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU

    @staticmethod
    def _compute_padding(in_size, kernel, stride, out_size):
        return kernel + (out_size - 1) * stride - in_size

    def reset_shape(self, input_shape, output_shape):
        self.output_shape = output_shape
        self.input_shape = input_shape
        padding = self.compute_padding(input_shape)
        self._set_multidimensional_attr('padding', padding, 2 * dim)

    @staticmethod
    def _out_size(in_size, stride):
        assert in_size % stride == 0
        return cdiv(in_size, stride)

    def _set_multidimensional_attr(self, attr, var, dim):
        if isinstance(var, tuple) or isinstance(var, list):
            assert (len(var) == dim), f'{attr} has dimension {len(var)}, expected dimension: {dim}'
            assert all((isinstance(v, int) for v in var)), f'all values must be ints'
            setattr(self, attr, tuple(var))
        else:
            assert isinstance(var, int), f'{attr} is of type: {type(var)}, expected one of: tuple, list, int'
            setattr(self, attr, tuple([var] * dim))

    def out_shape(self, in_shape=None):
        if in_shape is None:
            assert self.input_shape is not None
            in_shape = self.input_shape
        if self.output_shape is not None:
            return self.output_shape
        self.output_shape = tuple((ConvNormRelu._out_size(in_shape[i], self.stride[i]) for i in range(self.dim)))
        return self.output_shape

    def forward(self, x):
        if hasattr(self, 'check_shape') and self.check_shape:
            assert self.input_shape is None or x.shape == self.input_shape
        x = F.pad(x, self.padding)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvNormRelu2d(ConvNormRelu):

    def __init__(self, out_channels, in_channels=None, leaky=False, downsample=False, kernel=None, stride=None,
                 padding=None, input_shape=None, **kwargs):
        super(ConvNormRelu2d, self).__init__(leaky=leaky, downsample=downsample, kernel=kernel, stride=stride,
                                             padding=padding, input_shape=input_shape, dim=2, **kwargs)
        if in_channels is None:
            self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=self.kernel, stride=self.stride, padding=0)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel,
                                  stride=self.stride, padding=0)
        self.norm = nn.BatchNorm2d(out_channels)

    def compute_padding(self, shape):
        vertical_padding = self._compute_padding(shape[0], kernel=self.kernel[0], stride=self.stride[0],
                                                 out_size=self.out_shape(shape)[0])
        horizontal_padding = self._compute_padding(shape[1], kernel=self.kernel[1], stride=self.stride[1],
                                                 out_size=self.out_shape(shape)[1])
        pad_top, pad_left = vertical_padding // 2, horizontal_padding // 2
        pad_bottom, pad_right = vertical_padding - pad_top, horizontal_padding - pad_left
        self.padding = (pad_left, pad_right, pad_top, pad_bottom)
        return self.padding


class ConvNormRelu1d(ConvNormRelu):

    def __init__(self, out_channels, in_channels=None, leaky=False, downsample=False, kernel=None, stride=None,
                 padding=None, input_shape=None, **kwargs):
        super(ConvNormRelu1d, self).__init__(leaky=leaky, downsample=downsample, kernel=kernel, stride=stride,
                                             padding=padding, input_shape=input_shape, dim=1, **kwargs)
        if in_channels is None:
            self.conv = nn.LazyConv1d(out_channels=out_channels, kernel_size=self.kernel, stride=self.stride, padding=0)
        else:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel,
                                  stride=self.stride, padding=0)
        self.norm = nn.BatchNorm1d(out_channels)

    def compute_padding(self, shape):
        length = shape[0]
        padding = self._compute_padding(length, kernel=self.kernel[0], stride=self.stride[0],
                                        out_size=self.out_shape(shape)[0])
        pad_left = padding // 2
        pad_right = padding - pad_left
        self.padding = (pad_left, pad_right)
        return self.padding


def cdiv(a, b):
    # ceiling division
    return -(-int(a) // int(b))
