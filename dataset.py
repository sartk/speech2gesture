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
        return item['audio'].unsqueeze(0).float(), item['gesture'].permute(1, 0).float()

    def __len__(self):
        return len(self.dataset)

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
            audio_end = int(self.clip_duration * ((librosa.get_duration(filename=str(audio_file)) // self.clip_duration)
                                                  - 1))
            windows = range(0, audio_end, self.clip_duration // 2)
            print(self.save_dir / f'{name}_{len(windows) - 1}.pt')
            if (self.save_dir / f'{name}_{len(windows) - 1}.pt').is_file():
                continue
            exp_map = self.mocap_pipeline.apply(bvh_file)
            frame = 0
            frame_window, frame_step = self.gesture_fps * self.clip_duration, self.gesture_fps * self.clip_duration // 2
            for i, t in enumerate(windows):
                dest = self.save_dir / f'{name}_{i}.pt'
                if dest.is_file():
                    continue
                audio = self.mel_spec.apply(librosa.load(audio_file, offset=t, duration=self.clip_duration, mono=True))
                gesture = exp_map[frame:frame + frame_window, :]
                torch.save({'audio': torch.from_numpy(audio), 'gesture': torch.from_numpy(gesture)}, dest)
                dataset.append(dest)
                frame += frame_step
        return dataset

"""
for i, audio_file in enumerate(sorted(Path('C:/Users/Pinscreen/Dev/sarthak/speech2gesture/data/TrinityDataset/raw_data/train/Audio').iterdir())):
     src_bvh = str(audio_file).replace('Audio', 'Motion').replace('.wav', '.bvh')
     src_txt = str(audio_file).replace('Audio', 'Transcripts').replace('.wav', '.json')
     if i < 23:
             continue
     elif i < 28:
             tar = 'val'
     else:
             tar = 'test'
     tar_wav = str(audio_file).replace('train', tar)
     tar_bvh = src_bvh.replace('train', tar)
     tar_txt = src_txt.replace('train', tar)
     shutil.move(str(audio_file), tar_wav)
     shutil.move(src_bvh, tar_bvh)
     shutil.move(src_txt, tar_txt)
"""
