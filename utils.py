import os
import time
from argparse import ArgumentParser, ArgumentTypeError
from os.path import getctime
from pathlib import Path, PurePath
from typing import Iterable, Dict, Tuple, Callable
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

Loss = Callable[[Tensor, Tensor], Tensor]
DataLoaders = Dict[str, DataLoader]

try:
    os.environ['NV_GPU']
except:
    os.environ['NV_GPU'] = ''


def logger(path: Path, delimiter: str = ',') -> Callable[..., None]:
    """
    Creates a logger that prints to DIRECTORY. Also returns the current timestamp.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(*args):
        line = delimiter.join(map(str, args))
        with open(path, 'a+') as f:
            f.write(line)
            f.write('\n')
    return log


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def set_up_args() -> ArgumentParser:
    """
    Adds the hyper-parameter arguments to the parser
    """
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=str, default=os.environ['NV_GPU'], help='Comma separated list of CUDA visible GPUs')
    parser.add_argument('--lr_generator', type=float, default=1e-4)
    parser.add_argument('--lr_discriminator', type=float, default=1e-4)
    parser.add_argument('--lambda_d', type=float, default=1)
    parser.add_argument('--dataset', type=str, default='WavBVH', choices=['WavBVH', 'WavKeypoints'])
    parser.add_argument('--lambda_g', type=float, default=1)
    parser.add_argument('--experiments', type=Path, default='C:/Users/Pinscreen/Dev/sarthak/speech2gesture/experiments/')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--data', type=Path, default='C:/Users/Pinscreen/Dev/sarthak/speech2gesture/data/TrinityDataset/')
    parser.add_argument('--precision', type=str, default='full')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser


def latest(all_models: Iterable[Path]) -> Path:
    """
    Finds the latest checkpoint out of all the models.
    """
    return max(filter(lambda path: '_best.pt' in str(path), all_models), key=getctime)


def set_precision(tensor: Tensor, precision: str) -> Tensor:
    """
    Sets the precision of the input tensor.
    """
    if precision == 'float' or precision == 'full' or precision == 'float32':
        return tensor.float()
    elif precision == 'half' or precision == 'float16':
        return tensor.half()
    return tensor


def save_checkpoint(checkpoint, note):
    args = checkpoint['args']
    save_file = PurePath(f"{args.run_id}{'_' if note else ''}{note}.pt")
    path: Path = args.checkpoints_dir / save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return f'Latest model saved in {path}.'

