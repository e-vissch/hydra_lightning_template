from functools import partial
import pandas as pd
import torch
from src.datamodules.base_module import BaseDataModule
from torch.utils.data import DataLoader, IterableDataset


def get_data(input_df, float_dtypes, int_dtypes):
    pass


class RealDataset(IterableDataset):
    def __init__(
        self,
        data_getter,
        input_df,
        float_dtypes=torch.float32,
        int_dtypes=torch.int32,
        **data_getter_kwargs,
    ):
        self.input_df = input_df
        self.data_getter = data_getter
        self.dtypes = {"float_dtypes": float_dtypes, "int_dtypes": int_dtypes}
        self.data_getter_kwargs = data_getter_kwargs

    def __iter__(self):
        return self.data_getter(self.input_df, **self.data_getter_kwargs, **self.dtypes)


class RealDataModule(BaseDataModule):
    def __init__(
        self,
        task,
        data_getter,
        train_df,
        val_df,
        datamodule_config,
        loader_config,
    ) -> None:

        super().__init__(datamodule_config, task)
        self.train_dataset = RealDataset(
            data_getter, input_df=train_df, **datamodule_config.dataset
        )
        self.val_dataset = RealDataset(
            data_getter, input_df=val_df, **datamodule_config.dataset
        )
        self.loader_config = loader_config

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.loader_config)


def module_factory(
    data_getter,
    task_config,
    datamodule_config,
    loader_config,
):
    df = pd.read_csv(datamodule_config.dataset.pop("data_path"), sep="\t")

    train_df = df.sample(
        frac=datamodule_config.dataset.pop("train_frac"), random_state=42, axis=1
    )
    val_df = df.drop(train_df.columns, axis=1)

    return RealDataModule(
        data_getter=data_getter,
        train_df=train_df,
        val_df=val_df,
        task_config=task_config,
        datamodule_config=datamodule_config,
        loader_config=loader_config,
    )


registry = {"base": partial(module_factory, get_data)}
