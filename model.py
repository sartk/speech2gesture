import math
from collections import OrderedDict

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AudioToPose(nn.Module):

    def __init__(self, pose_shape: Tuple[int, int], input_shape: Tuple[int, int], encoder_dim=2):
        super(AudioToPose, self).__init__()
        pose_dof, frames = pose_shape
        h, w = input_shape
        self.encoder_dim = encoder_dim
        self.audio_encoder = self.get_audio_encoder(encoder_dim, h, w)
        if encoder_dim == 2:
            self.resize = lambda t: torch.squeeze(
                F.interpolate(t, size=(frames, 1), mode='bilinear', align_corners=False), dim=-1)
        elif encoder_dim == 1:
            self.resize = lambda t: F.interpolate(t, size=(frames,), mode='bilinear', align_corners=False)
        self.unet_encoder_labels = [5, 6, 7, 8, 9, 10]
        self.unet_encoder = nn.ModuleList([
            nn.Sequential(
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(frames,)),
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(frames,))),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                           input_shape=(frames,), output_shape=(cdiv(frames, 2),)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                             input_shape=(cdiv(frames, 2),), output_shape=(cdiv(frames, 4),)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                            input_shape=(cdiv(frames, 4),), output_shape=(cdiv(frames, 8),)),
            ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                           input_shape=(cdiv(frames, 8),), output_shape=(cdiv(frames, 16),)),
            nn.Sequential(ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                                                         input_shape=(cdiv(frames, 16),),
                                                         output_shape=(cdiv(frames, 32),)),
                                          nn.Upsample(size=(cdiv(frames, 16),)))
        ])
        self.unet_decoder_labels = [9, 8, 7, 6, 5]
        self.unet_decoder = nn.ModuleList([
            nn.Sequential(
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

    def get_audio_encoder(self, d, h, w):
        if d == 2:
            return nn.ModuleList([
                ConvNormRelu2d(in_channels=1, out_channels=64, leaky=True, downsample=False, input_shape=(h, w)),
                ConvNormRelu2d(in_channels=64, out_channels=64, leaky=True, downsample=True, input_shape=(h, w),
                               output_shape=(cdiv(h, 2), cdiv(w, 2))),
                ConvNormRelu2d(in_channels=64, out_channels=128, leaky=True, downsample=False,
                               input_shape=(cdiv(h, 2), cdiv(w, 2))),
                ConvNormRelu2d(in_channels=128, out_channels=128, leaky=True, downsample=True,
                               input_shape=(cdiv(h, 2), cdiv(w, 2)),
                               output_shape=(cdiv(h, 4), cdiv(w, 4))),
                ConvNormRelu2d(in_channels=128, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(h, 4), cdiv(w, 4))),
                ConvNormRelu2d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                               input_shape=(cdiv(h, 4), cdiv(w, 4)),
                               output_shape=(cdiv(h, 8), cdiv(w, 8))),
                ConvNormRelu2d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(h, 8), cdiv(w, 8))),
                ConvNormRelu2d(in_channels=256, out_channels=256, leaky=True, downsample=False, kernel=(3, 8), stride=1,
                               padding=0, input_shape=(cdiv(h, 8), cdiv(w, 8)))
            ])
        elif d == 1:
            return nn.ModuleList([
                ConvNormRelu1d(in_channels=81, out_channels=64, leaky=True, downsample=False, input_shape=(w,)),
                ConvNormRelu1d(in_channels=64, out_channels=64, leaky=True, downsample=True, input_shape=(w,),
                               output_shape=(cdiv(w, 2),)),
                ConvNormRelu1d(in_channels=64, out_channels=128, leaky=True, downsample=False,
                               input_shape=(cdiv(w, 2),)),
                ConvNormRelu1d(in_channels=128, out_channels=128, leaky=True, downsample=True,
                               input_shape=(cdiv(w, 2),),
                               output_shape=(cdiv(w, 4),)),
                ConvNormRelu1d(in_channels=128, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(w, 4),)),
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=True,
                               input_shape=(cdiv(w, 4),),
                               output_shape=(cdiv(w, 8),)),
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False,
                               input_shape=(cdiv(w, 8),)),
                ConvNormRelu1d(in_channels=256, out_channels=256, leaky=True, downsample=False, kernel=(8,),
                               stride=1,
                               padding=0, input_shape=(cdiv(w, 8),))
            ])


        def weight_init(m):
            if m == self.logits[-1]:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

        self.apply(weight_init)

    def set_input_shape(self, shape: Tuple[int, int]):
        for conv_block in self.audio_encoder:
            conv_block.compute_padding(shape)
            shape = conv_block.out_shape(shape)

    def forward(self, x: Tensor) -> Tensor:
        if self.encoder_dim == 2:
            x = x.unsqueeze(1)
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
