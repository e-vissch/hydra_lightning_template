
from dataclasses import dataclass, field
from os import name
from typing import Literal, Optional, Union

from numpy import save
from omegaconf import MISSING

INTERPOLATED_FROM_PARENT = MISSING


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class CosineSchedulerConfig:
    name: str = "cosine"
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class TrainerConfig:
    devices: Union[int, str] = 1
    accelerator: str = "gpu"
    max_epochs: int = 200
    val_check_interval: Union[float, int] = 10  # val every n epochs
    gradient_clip_val: Optional[float] = None # Note default used to be 0 but they changed
    log_every_n_steps: int = 10
    limit_train_batches: float = (
        1.0  # train on full dataset, can be used to toggle quick run
    )
    limit_val_batches: Union[float, int] = 2


@dataclass
class LoaderConfig:
    batch_size: int = INTERPOLATED_FROM_PARENT


@dataclass
class WandBConfig:
    project: str = "dna"
    group: str = ""
    job_type: str = "training"
    mode: str = "online"
    name: Optional[str] = None
    save_dir: str = "."
    id: Optional[str] = None


@dataclass
class CheckpointConfig:
    monitor: str = "mse"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    auto_insert_metric_name: bool = False
    verbose: bool = True


@dataclass
class ModelConfig:
    pass


@dataclass
class DataModuleConfig:
    name: str = "datamodule"


@dataclass
class TaskConfig:
    ...


@dataclass
class TrainConfig:
    name: str = "train" # Literal["train", "pretrained"] 
    saved_model: Optional[str] = None 


def validate_config(config: "Config"):
    # WRITE VALIDATION LOGIC HERE
    pass


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[CosineSchedulerConfig] = field(
        default_factory=CosineSchedulerConfig
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model_checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)

    wandb: Optional[WandBConfig] = field(default_factory=WandBConfig)


    def __post_init__(self):
        # validation has to done here because of the way hydra works
        validate_config(self)
