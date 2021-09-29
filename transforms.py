import math
from pathlib import Path

import librosa
import numpy as np
import scipy.signal

from pymo.data import MocapData
from pymo.parsers import BVHParser
import transforms3d as t3d


class Transform:

    def __init__(self, kwargs=None):
        self.kwargs = dict() if kwargs is None else kwargs

    def apply(self, x):
        return self.identity(x)

    def invert(self, y):
        return self.identity(y)

    def identity(self, x):
        return x


class MocapDataToExpMap(Transform):

    def __init__(self, kwargs=None):
        super().__init__(kwargs)

    def apply(self, mocap_data: MocapData, use_deg: bool = True):
        df = mocap_data.values
        frames = np.zeros(df.shape)
        for frame_number, (time_delta, channels) in enumerate(df.iterrows()):
            channels = np.asarray(channels)
            channel_num = 0
            while channel_num < 3:
                frames[frame_number, channel_num] = channels[channel_num]
                channel_num += 1
            for z, x, y in [channels[i:i + 3] for i in range(3, len(channels), 3)]:
                if use_deg:
                    z, x, y = math.radians(z), math.radians(x), math.radians(y)
                vec, theta = t3d.euler.euler2axangle(z, x, y, 'rzxy')
                scaled = vec * theta
                for i in range(3):
                    frames[frame_number, channel_num] = scaled[i]
                    channel_num += 1
        return frames

    def invert(self, exp_map, back_to_deg=True):
        frames = np.zeros(exp_map.shape)
        assert exp_map.shape[1] == 174
        for i in range(exp_map.shape[0]):
            for j in range(3):
                frames[i, j] = exp_map[i, j]
            for j in range(3, exp_map.shape[1], 3):
                scaled = np.array([exp_map[i, j], exp_map[i, j + 1], exp_map[i, j + 2]])
                theta = np.linalg.norm(scaled)
                if theta == 0:
                    z, x, y = 0., 0., 0.
                else:
                    vector = scaled / theta
                    z, x, y = t3d.euler.axangle2euler(vector, theta, 'rzxy')
                if back_to_deg:
                    z, x, y = math.degrees(z), math.degrees(x), math.degrees(y)
                frames[i, j], frames[i, j + 1], frames[i, j + 2] = z, x, y
        return frames


class BVHtoMocapData(Transform):

    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        self.header = ''
        self.parser = BVHParser()

    def apply(self, bvh_file: Path) -> MocapData:
        with open(bvh_file, 'r') as f:
            data = f.read()
            self.header = data.split('MOTION')[0]
        return self.parser.parse(bvh_file)

    def invert(self, mocap_data):
        content = self.header + 'MOTION\n' + f'Frames: {mocap_data.shape[0]}\n' + 'Frame Time:     0.0166667\n'
        for i in range(mocap_data.shape[0]):
            content += ' '.join(map(str, mocap_data[i].tolist())) + '\n'
        return content


class Pipeline:

    def __init__(self, transforms=(Transform,), kwargs=None):
        if kwargs is None:
            kwargs = [None] * len(transforms)
        self.transforms = [transform(kwarg) for transform, kwarg in zip(transforms, kwargs)]

    def apply(self, x):
        for f in self.transforms:
            x = f.apply(x)
        return x

    def invert(self, x):
        for f in reversed(self.transforms):
            x = f.invert(x)
        return x


class AudioToLogMelSpec(Transform):

    def __init__(self, kwargs=None):
        default = {'window_length': 0.1, 'dim': 64}
        super().__init__(default.update(kwargs) if kwargs is not None else default)

    def apply(self, input):
        audio, sample_rate = input
        spectr = librosa.feature.melspectrogram(audio, sr=sample_rate, window=scipy.signal.hanning,
                                                # win_length=int(window_length * sample_rate),
                                                hop_length=int(self.kwargs['window_length'] * sample_rate / 2),
                                                fmax=7500, fmin=125, n_mels=self.kwargs['dim'])
        eps = 1e-6
        log_spectr = np.log(abs(spectr) + eps)
        return np.transpose(log_spectr)
