import pprint
from collections import defaultdict
from typing import DefaultDict, Union, Callable, Dict

from torch import Tensor

from train import Trainer

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
                self.trainer.writer.add_scalar(f'{self.phase}Batch/{name}', value, global_step=self.global_step)
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
                self.trainer.writer.add_scalar(f'{self.phase}Epoch/{name}_{info}', value, global_step=self.epoch)
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
