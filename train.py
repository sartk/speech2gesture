import random
import time
from argparse import Namespace
from pathlib import Path, PurePath
from typing import DefaultDict, Union, Callable, Dict

import torch
from torch import nn, Tensor
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
from dataset import WavBVHDataset
from model import AudioToPose, PoseDiscriminator

import pprint
from collections import defaultdict

from transforms import Pipeline, MocapDataToExpMap, BVHtoMocapData


class Trainer:

    def __init__(self, checkpoint: Dict, args: Namespace):
        """
        Initial setup for training.
        """
        torch.manual_seed(0)
        random.seed(0)
        self.checkpoint = checkpoint
        self.args = args
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.check_required_args()
        self.data, shapes = self.get_data()
        self.generator = AudioToPose(input_shape=shapes[0], pose_shape=shapes[1])
        self.discriminator = PoseDiscriminator(pose_shape=shapes[1])
        self.generator.float()
        self.discriminator.float()
        self.mocap_pipeline = Pipeline([BVHtoMocapData, MocapDataToExpMap])
        with open('mocap_header.txt', 'r') as f:
            self.mocap_pipeline.transforms[0].header = f.read()
        self.loss = Trainer.get_losses()
        self.optim = self.get_optimizers()
        self.metric = self.get_metric_collectors()
        self.experiment_dir: Path = args.experiments / self.timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.experiment_dir / PurePath('tensorboard')
        self.checkpoints_dir = self.experiment_dir / PurePath('checkpoints')
        self.samples_dir = self.experiment_dir / PurePath('samples')
        self.tensorboard_dir.mkdir()
        self.checkpoints_dir.mkdir()
        self.log_file = self.experiment_dir / 'log.txt'
        self.log = utils.logger(self.log_file)
        self.best_checkpoint: Union[Path, None] = None
        self.checkpoint = dict()

    def check_required_args(self) -> None:
        """
        Checks if self.args namespace contains all the required args
        """
        args = ['lr_generator', 'lr_discriminator', 'lambda_d', 'lambda_g', 'experiments', 'epochs', 'patience',
                'gpu', 'precision', 'dataset', 'data', 'num_workers', 'batch_size']
        for arg in args:
            assert hasattr(self.args, arg), f'Error, missing arg: {arg}'

    @staticmethod
    def get_losses():
        """
        Returns a namespace of required loss functions.
        """
        losses = Namespace()
        losses.l1 = nn.L1Loss()
        losses.mse = nn.MSELoss()
        return losses

    def get_optimizers(self):
        """
        Returns a namespace of optimizers.
        """
        optim = Namespace()
        optim.generator = Adam(params=self.generator.parameters(), lr=self.args.lr_generator)
        optim.discriminator = Adam(params=self.discriminator.parameters(), lr=self.args.lr_discriminator)
        return optim

    def get_metric_collectors(self):
        """
        Returns a namespace of Metric Collectors.
        """
        metric = Namespace()
        metric.train = MetricCollector(phase='train', patience_monitor=lambda summary: summary['generator_loss'],
                                       trainer=self)
        metric.val = MetricCollector(phase='val', patience_monitor=lambda summary: summary['generator_loss'],
                                       trainer=self)
        return metric

    def get_data(self):
        """
        Returns  the data.
        """
        dataloader = Namespace()
        shapes = None
        if self.args.dataset == 'WavBVH':
            source = WavBVHDataset
        else:
            source = None
        for mode in ['train', 'val', 'test']:
            dataset = WavBVHDataset(self.args.data, group=mode)
            setattr(dataloader, mode, DataLoader(dataset=dataset,
                                                 num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                                 shuffle=True))
            if shapes is None:
                shapes = [item.shape[-2:] for item in dataset[0]]
        return dataloader, shapes

    def loop(self, audio, real_pose, mode='train'):
        """
        A single batch cycle.
        """
        # Run Model
        pred_pose = self.generator(audio)
        # Update Discriminator
        discriminator_pred = self.discriminator(pred_pose.detach())
        discriminator_real = self.discriminator(real_pose)
        self.optim.discriminator.zero_grad()
        real_pose_loss = self.loss.mse(torch.ones_like(discriminator_real), discriminator_real)
        fake_pose_loss = self.loss.mse(torch.zeros_like(discriminator_pred), discriminator_pred)
        discriminator_loss = real_pose_loss + fake_pose_loss
        if mode == 'train':
            discriminator_loss.backward()
            self.optim.discriminator.step()
        # Update Generator
        self.optim.generator.zero_grad()
        l1_loss = self.loss.l1(pred_pose, real_pose)
        discriminator_pred = self.discriminator(pred_pose)
        adversarial_loss = self.loss.mse(torch.ones_like(discriminator_pred), discriminator_pred)
        generator_loss = l1_loss + adversarial_loss
        if mode == 'train':
            generator_loss.backward()
            self.optim.generator.step()
        # Save Results
        results = {
                'generator_l1_loss': l1_loss,
                'generator_adversarial_loss': adversarial_loss,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss
        }
        return results, pred_pose

    def save_checkpoint(self, note='', path: Path = None, best=False) -> Path:
        """
        Saves the current checkpoint (in checkpoints dir by default).
        """
        save_file = self.experiment_dir / (note + '.pt')
        if path is None:
            path: Path = self.checkpoints_dir / save_file
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)
        if best:
            self.best_checkpoint = path
        return path

    def device(self, x: Tensor) -> Tensor:
        """
        Moves a tensor to the right device.
        """
        if self.args.gpu:
            x = x.cuda()
        if self.args.precision == 'half':
            x = x.half()
        return x

    def run(self):
        """
        Runs the training loop.
        """
        for epoch in tqdm(range(self.args.epochs), desc='epochs', position=0):
            samples_dir = self.samples_dir / f'Epoch_{epoch}'
            samples_dir.mkdir(parents=True, exist_ok=True)
            if self.metric.val.patience >= self.args.patience:
                return
            self.generator.train()
            self.discriminator.train()
            for audio, pose in tqdm(self.data.train, desc='batch', leave=False, position=1):
                audio, pose = self.device(audio), self.device(pose)
                results, _ = self.loop(audio, pose, 'train')
                self.metric.train.update(results)
            self.metric.train.epoch_step()
            self.generator.eval()
            self.discriminator.eval()
            for i, (audio, pose) in enumerate(tqdm(self.data.val, desc='batch', leave=False, position=1)):
                audio, pose = self.device(audio), self.device(pose)
                results, pred_pose = self.loop(audio, pose, 'val')
                if i < 10:
                    bvh = self.mocap_pipeline.invert(pred_pose[0].permute(1, 0).numpy())
                    with open(samples_dir / f'Sample_{i}.bvh', 'w+') as f:
                        f.write(bvh)
                self.metric.val.update(results)
            self.metric.val.epoch_step()
            self.checkpoint.update({
                'model_state_dict': {
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict()
                },
                'optimizer_state_dict': {
                    'generator': self.optim.generator.state_dict(),
                    'discriminator': self.optim.discriminator.state_dict()
                }
            })



Number = Union[float, int]
TorchNumber = Union[Number, Tensor]
Loss = Callable[[Tensor, Tensor], Tensor]


class MetricCollector:

    def __init__(self, phase: str, patience_monitor: Callable[[Dict], Number] = lambda x: None, trainer: Trainer = None,
                 epoch_init: int = 0):
        """
        Initializer for the MetricCollector class.
        """
        # keeps track of sums of metrics across all epochs
        self.r1: DefaultDict[int, DefaultDict[str, Number]] = defaultdict(lambda: defaultdict(lambda: 0.))
        # keeps track of sums of squared metrics across all epochs
        self.r2: DefaultDict[int, DefaultDict[str, Number]] = defaultdict(lambda: defaultdict(lambda: 0.))
        # keeps track of sample counts across all epochs
        self.samples: DefaultDict[int, int] = defaultdict(lambda: 0)
        self.phase: str = phase
        self.trainer: Trainer = trainer
        # function used that takes in the summary and creates a criteria used for stopping
        self.patience_monitor: Callable[[int], Number] = lambda summary: patience_monitor(summary['mean'])
        self.best_epoch = 0
        self.lowest_patience_metric = float('inf')
        self.patience: int = 0
        self.epoch: int = epoch_init
        self.global_step: int = 0
        self.summary_cache: Dict[int, Dict] = dict()

    def update(self, new_values: Dict, n=1) -> None:
        """
        Updates the current epoch with new values for the metrics.
        """
        for name, value in new_values.items():
            if self.trainer:
                pass
                #self.trainer.writer.add_scalar(f'{self.phase}Batch/{name}', value, global_step=self.global_step)
            if isinstance(value, Tensor):
                value = value.item()
            self.r1[self.epoch][name] += value * n
            self.r2[self.epoch][name] += (value ** 2) * n
        self.samples[self.epoch] += n
        self.global_step += 1

    def summary(self, epoch=None) -> Dict[str, Union[Number, Dict[str, Number]]]:
        """
        Generates a summary (means and SDs of all metrics) on given epoch (or current if not specified).
        """
        if epoch is None:
            epoch = self.epoch
        if epoch in self.summary_cache.keys():
            return self.summary_cache[epoch]
        means = {name: self.r1[epoch][name] / self.samples[epoch]
                 for name in self.r1[epoch]}
        sds = {name: ((self.r2[epoch][name] / self.samples[epoch]) - (means[name] ** 2))
               for name in self.r2[epoch]}
        summary = {'epoch': epoch, 'mean': means, 'sd': sds}
        self.summary_cache[epoch] = summary
        return summary

    def epoch_step(self) -> None:
        """
        Steps epoch, computes summary, saves checkpoint, tracks patience.
        """
        summary = self.summary()
        for info in ['mean', 'sd']:
            for name, value in summary[info].items():
                pass
                #self.trainer.writer.add_scalar(f'{self.phase}Epoch/{name}_{info}', value, global_step=self.epoch)
        if 'val' in self.phase.lower():
            metric = self.patience_monitor(summary)
            if metric < self.lowest_patience_metric:
                self.lowest_patience_metric = metric
                self.patience = 0
                self.best_epoch = self.epoch
                self.trainer.checkpoint['summary'] = summary
                self.trainer.save_checkpoint('best', best=True)
            else:
                self.patience += 1
        summary['patience'] = self.patience
        self.trainer.log(f'{self.phase}: ' + pprint.pformat(summary, indent=4))
        self.epoch += 1

    def best_summary(self):
        return self.summary(self.best_epoch)

    def reset_patience(self):
        self.lowest_patience_metric = float('inf')
        self.patience = 0
