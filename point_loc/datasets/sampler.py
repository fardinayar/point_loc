from point_loc.registry import DATA_SAMPLERS
import itertools
import math
from typing import Iterator, Optional, Sized
import numpy as np
import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed


@DATA_SAMPLERS.register_module()
class WeightedTargetSampler(Sampler):
    """A sampler that returns data with higher target values more frequently for multi-output regression tasks.

    Args:
        dataset (Sized): The dataset.
        target_column (int): The index of the target column to use for weighting.
        temperature (float): Controls the strength of the weighting. Higher values increase the bias towards higher target values.
        shuffle (bool): Whether to shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 temperature: float = 0.1,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.temperature = temperature
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        # Calculate weights based on target values
        self.weights = self._calculate_weights()

    def _calculate_weights(self):
        targets = [self.dataset[i]['data_samples'].gt_values.abs().sum() for i in range(len(self.dataset))]
        targets = np.array(targets)
        weights = targets
        return weights / weights.sum()

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.multinomial(
                torch.tensor(self.weights),
                num_samples=self.total_size,
                replacement=True,
                generator=g
            ).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch