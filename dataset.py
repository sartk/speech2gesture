from pathlib import Path
from typing import Dict, Tuple, List, Union
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transforms import BVHtoMocapData, MocapDataToExpMap, Pipeline, AudioToLogMelSpec
import librosa
from tqdm import tqdm
import pickle

Number = Union[float, int]


class WavBVHDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        item = torch.load(self.dataset[index])
        return item['audio'], item['gesture']

    def __init__(self, dataset: Path, gesture_fps=60, clip_duration=4, group='train', transcripts=False):
        self.audio_dir: Path = dataset / 'raw_data' / group / 'Audio'
        self.motion_dir: Path = dataset / 'raw_data' / group / 'Motion'
        self.save_dir: Path = dataset / group / 'processed_data'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path: Path = dataset / 'info.pickle'
        self.gesture_fps: Number = gesture_fps
        self.clip_duration: int = clip_duration
        self.mocap_pipeline = Pipeline([BVHtoMocapData, MocapDataToExpMap])
        self.mel_spec = AudioToLogMelSpec()
        if self.dataset_path.is_file():
            with open(self.dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset: List[Path] = self.build_dataset()
            with open(self.dataset_path, 'wb') as f:
                pickle.dump(self.dataset, f)

    def build_dataset(self):
        dataset = []
        for audio_file in tqdm(list(self.audio_dir.iterdir())):
            name = audio_file.name.split('.')[0]
            bvh_file = self.motion_dir / (name + '.bvh')
            if not audio_file.is_file() or not bvh_file.is_file():
                print(f'not found: {audio_file}')
                continue
            exp_map = self.mocap_pipeline.apply(bvh_file)
            frame = 0
            audio_end = int(self.clip_duration * (librosa.get_duration(filename=str(audio_file)) // self.clip_duration))
            frame_window, frame_step = self.gesture_fps * self.clip_duration, self.gesture_fps * self.clip_duration // 2
            for i, t in enumerate(range(0, audio_end, self.clip_duration // 2)):
                dest = self.save_dir / f'{name}_{i}.pt'
                if dest.is_file():
                    continue
                audio = self.mel_spec.apply(librosa.load(audio_file, offset=t, duration=self.clip_duration, mono=True))
                gesture = exp_map[frame:frame + frame_window, :]
                torch.save({'audio': torch.from_numpy(audio), 'gesture': torch.from_numpy(gesture)}, dest)
                dataset.append(dest)
                frame += frame_step
        return dataset

