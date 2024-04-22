import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.train_object import TrainObject
from src.utils.config import get_object_from_registry, process_config, register_configs
from src.datamodules import datamodules_registry

def setup_wandb(config: DictConfig):
    # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
    # Can pass in config_exclude_keys='wandb' to remove certain groups
    import wandb
    from wandb.errors import CommError

    get_logger = lambda final_conf: WandbLogger(
        config=OmegaConf.to_container(final_conf, resolve=True),
        settings=wandb.Settings(start_method="fork"),  # type: ignore
        **final_conf.wandb,
    )

    try:
        return get_logger(config)

    except CommError:
        config.wandb.mode = "offline"
        return get_logger(config)



def add_default_callbacks(config: DictConfig, trainer_dict: DictConfig):
    callbacks = trainer_dict.get("callbacks", [])
    callbacks.append(ModelCheckpoint(**config.model_checkpoint))
    callbacks.append(LearningRateMonitor(**config.learning_rate_monitor))
    return trainer_dict | {"callbacks": callbacks}


def get_trainer(config: DictConfig):
    logger = setup_wandb(config) if config.get("wandb") is not None else None
    trainer_dict = add_default_callbacks(config.callbacks, config.trainer)
    return L.Trainer(**trainer_dict, logger=logger)


def get_data(task, datamodule_config: DictConfig, loader_config):
    datamodule_getter, datamodule_config_copy = get_object_from_registry(
        datamodule_config, datamodules_registry, init=False
    )
    return datamodule_getter(task, datamodule_config_copy, loader_config)


def train_from_config(config):
    trainer = get_trainer(config)
    model = TrainObject(config) # lightning wrapper around model, includes training logic
    datamodule = get_data(model.task, config.datamodule)
    trainer.fit(model, datamodule)


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: DictConfig):
    config = process_config(config)
    train_from_config(config)


if __name__ == "__main__":
    register_configs()
    main()
