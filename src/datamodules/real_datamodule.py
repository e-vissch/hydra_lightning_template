import torch
from src.datamodules.base_module import BaseDataModule
from torch.utils.data import DataLoader, IterableDataset


class RealDataset(IterableDataset):
    def __init__(
        self,
        data_getter,
        input_list,
        float_dtypes=torch.float32,
        int_dtypes=torch.int32,
        **data_getter_kwargs,
    ):
        self.data_getter = data_getter

        self.input_list = input_list
        self.dtypes = {"float_dtypes": float_dtypes, "int_dtypes": int_dtypes}
        self.data_getter_kwargs = data_getter_kwargs

    def __iter__(self):
        for args in self.input_list:
            yield self.data_getter(*args, **self.dtypes, **self.data_getter_kwargs)


class RealDataModule(BaseDataModule):
    def __init__(
        self,
        data_getter,
        task,
        train_list,
        val_list,
        datamodule_config,
        loader_config,
    ) -> None:
        super().__init__(datamodule_config)
        self.train_dataset = RealDataset(
            data_getter, input_list=train_list, **datamodule_config.dataset
        )
        self.val_dataset = RealDataset(
            data_getter, input_list=val_list, **datamodule_config.dataset
        )
        self.loader_config = loader_config

        self.task = task

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.loader_config)
