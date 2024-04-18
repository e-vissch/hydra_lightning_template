from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import partial
from typing import Union
from unittest.mock import Base

import numpy as np
from sklearn import base
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.datamodules.base_module import BaseDataModule
from src.tasks.base import BaseTask


@dataclass
class BaseSampler(abc.ABC):
    task: BaseTask

    def sample(self):
        return NotImplementedError

    def sample_torch(
        self,
        train_type="train",
        float_dtypes=torch.float32,
        int_dtypes=torch.int32,
    ):
        pos_feats, inputs, targets, dynamic_kwargs = self.sample(
            train_type
        )

        return (
            (torch.tensor(np.stack(pos_feats, axis=-1), dtype=int_dtypes),
            torch.tensor(np.stack(inputs, axis=-1), dtype=float_dtypes)),
            torch.tensor(np.stack(targets, axis=-1), dtype=float_dtypes),
            {key: torch.tensor(val) for key, val in dynamic_kwargs.items()},
        )



class SharedProbDataset(IterableDataset):
    def __init__(
        self,
        sampler: BaseSampler,
        float_dtypes=torch.float32,
        int_dtypes=torch.int32,
        sampler_kwargs=None,
        n_batches=100,
    ):
        # because dataset is infinite, we need to specify how many batches to sample
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.dtypes = {"float_dtypes": float_dtypes, "int_dtypes": int_dtypes}
        self.n_batches = n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            yield self.sampler.sample_torch(**self.dtypes, **self.sampler_kwargs)



class ProbDataModule(BaseDataModule):
    def __init__(self, sampler_factory, task, datamodule_config, loader_config) -> None:
        super().__init__(datamodule_config, task)

        dataset_config = datamodule_config.dataset.copy()
        n_batches = dataset_config.pop("n_batches")

        self.sampler: BaseSampler = sampler_factory(task, dataset_config)

        self.train_dataset = SharedProbDataset(self.sampler, n_batches=n_batches)
        self.val_dataset = SharedProbDataset(
            self.sampler, sampler_kwargs={"train_type": "val"}, n_batches=n_batches
        )
        self.loader_config = loader_config

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.loader_config)



sim_registry = {"example": partial(ProbDataModule, BaseSampler)}