import random
import time
from argparse import Namespace
from pathlib import Path, PurePath
from typing import Dict, Union

import torch
from torch import nn, Tensor
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
from dataset import WavBVHDataset
from model import AudioToPose, PoseDiscriminator
from metrics import MetricCollector


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
        self.loss = Trainer.get_losses()
        self.optim = self.get_optimizers()
        self.metric = self.get_metric_collectors()
        self.experiment_dir: Path = args.experiments / self.timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.experiment_dir / PurePath('tensorboard')
        self.checkpoints_dir = self.experiment_dir / PurePath('checkpoints')
        self.tensorboard_dir.mkdir()
        self.checkpoints_dir.mkdir()
        self.log_file = args.experiments_dir / 'log.txt'
        self.log = utils.logger(self.log_file)
        self.best_checkpoint: Union[Path, None] = None
        self.checkpoint = dict()

    def check_required_args(self) -> None:
        """
        Checks if self.args namespace contains all the required args
        """
        args = ['lr_generator', 'lr_discriminator', 'lambda_d', 'lambda_g', 'experiments', 'epochs', 'patience',
                'gpu', 'precision']
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
            source = WavKeypointsDataset
        for mode in ['train', 'val', 'test']:
            dataset = source(self.args.dataset, group=mode)
            setattr(dataloader, mode, DataLoader(dataset=dataset,
                                                 num_workers=self.args.num_workers, batch_size=self.args.batch_size))
            if shapes is None:
                shapes = [item.shape[-2:] for item in dataset[0]]
        return dataloader, shapes

    def loop(self, audio, real_pose, mode='train'):
        """
        A single batch cycle.
        """
        # Run Model
        pred_pose = self.generator(audio)
        discriminator_pred = self.discriminator(pred_pose)
        discriminator_real = self.discriminator(real_pose)
        # Update Discriminator
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
        adversarial_loss = self.loss.mse(torch.ones_like(discriminator_pred), discriminator_pred)
        generator_loss = l1_loss + adversarial_loss
        if mode == 'train':
            generator_loss.backward()
            self.optim.generator.step()
        # Save Results
        results = {
                'generator_l1_loss': l1_loss,
                'generator_adversarial_loss': adversarial_loss,
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
            for audio, pose in tqdm(self.data.val, desc='batch', leave=False, position=1):
                audio, pose = self.device(audio), self.device(pose)
                results, pred_pose = self.loop(audio, pose, 'val')
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
