import logging
import warnings
from typing import Any

import configs
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

from configs.schemas import Config

def get_object_from_registry(config, registry, init=True, attr_ls = []):
    config_copy = config.copy()

    to_pop = config_copy
    for attr in attr_ls:
        to_pop = getattr(to_pop, attr)

    if init:
        return registry[to_pop.pop("name")](**config_copy)

    return registry[to_pop.pop("name")], config_copy


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def allow_mutable_config_with_pop(config: DictConfig | Any):
    if isinstance(config, DictConfig):
        # Set struct mode to False for the current config
        OmegaConf.set_struct(config, False)
        # Recursively apply for all nested configs
        for key, value in config.items():
            allow_mutable_config_with_pop(value)


def process_config(config: DictConfig) -> DictConfig:  #
    # validate config, but want as dict, probably a better way to do this?
    _config: configs.schemas.Config = OmegaConf.to_object(config)

    log = get_logger()

    # enable adding new keys to config
    allow_mutable_config_with_pop(config)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    return config


def register_configs() -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("len", len)
    cs = ConfigStore.instance()
    cs.store(
        name="base_config", # used in config.yaml
        node=Config,
    )
