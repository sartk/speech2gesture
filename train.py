import random
import time
from argparse import Namespace
from pathlib import Path, PurePath
from typing import Dict, Union

import torch
from torch import nn, Tensor
from torch.optim import Adam
from tqdm import tqdm

import utils
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
        self.data, input_shape, pose_shape = self.get_dataloaders()
        self.generator = AudioToPose(input_shape=input_shape, pose_shape=pose_shape)
        self.discriminator = PoseDiscriminator(pose_shape=pose_shape)
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
        losses = Namespace()
        losses.l1 = nn.L1Loss()
        losses.mse = nn.MSELoss()
        return losses

    def get_optimizers(self):
        optim = Namespace()
        optim.generator = Adam(params=self.generator.parameters(), lr=self.args.lr_generator)
        optim.discriminator = Adam(params=self.discriminator.parameters(), lr=self.args.lr_discriminator)
        return optim

    def get_metric_collectors(self):
        metric = Namespace()
        metric.train = MetricCollector(phase='train', patience_monitor=lambda summary: summary['generator_loss'],
                                       trainer=self)
        metric.val = MetricCollector(phase='val', patience_monitor=lambda summary: summary['generator_loss'],
                                       trainer=self)
        return metric

    def get_dataloaders(self):
        return ..., ..., ...

    def loop(self, audio, real_pose, mode='train'):

        if mode == 'train':
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()

        pred_pose = self.generator(audio)
        discriminator_pred = self.discriminator(pred_pose)
        discriminator_real = self.discriminator(real_pose)

        discriminator_loss = self.loss.mse(torch.ones_like(discriminator_real), discriminator_real) + \
                             self.args.lambda_d * self.loss.mse(torch.zeros_like(discriminator_pred),
                                                                discriminator_pred)

        l1_loss = self.loss.l1(pred_pose, real_pose)
        adversarial_loss = self.loss.mse(torch.ones_like(discriminator_pred), discriminator_pred)
        generator_loss = l1_loss + adversarial_loss

        self.optim.discriminator.zero_grad()
        discriminator_loss.backward()

        self.optim.generator.zero_grad()
        generator_loss.backward()

        self.optim.discriminator.step()
        self.optim.generator.step()

        results = {
                'l1_loss': l1_loss,
                'adversarial_loss': adversarial_loss,
                'discriminator_loss': discriminator_loss
        }

        return results

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
        if self.args.gpu:
            x = x.cuda()
        if self.args.precision == 'half':
            x = x.half()
        return x

    def run(self):
        for epoch in tqdm(range(self.args.epochs), desc='epochs', position=0):
            if self.metric.val.patience >= self.args.patience:
                return
            for audio, pose in tqdm(self.data.train, desc='batch', leave=False, position=1):
                audio, pose = self.device(audio)
                results = self.loop(audio, pose, 'train')
                self.metric.train.update(results)
            self.metric.train.epoch_step()
            for audio, pose in tqdm(self.data.val, desc='batch', leave=False, position=1):
                results = self.loop(audio, pose, 'val')
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
