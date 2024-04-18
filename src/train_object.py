from collections import defaultdict

import lightning as L
import torch

from src.models.example_model import ExampleModel
from src.utils.config import get_logger, get_object_from_registry
from src.tasks.base import task_registry

scheduler_registry = {}


log = get_logger(__name__)


def model_from_checkpoint(config, _):
    trained_model = TrainObject.load_from_checkpoint(config.saved_model)
    return trained_model.model

def model_from_config(config, task):
    decoder = task.decoder_cls(**config.model.decoder)
    model = ExampleModel(config.model, decoder)
    return model


train_type_registry = {
    "train": model_from_config, #config.model
    "pretrained": model_from_checkpoint,
}



class TrainObject(L.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch.set_float32_matmul_precision("high")
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()

        self.task = get_object_from_registry(config.task, task_registry)
        self.model= train_type_registry[config.train_type](config, self.task)
        
        self.save_hyperparameters(config, logger=False)
        # Passing in config expands it one level, so can access by
        # self.hparams.train instead of self.hparams.config.train

    def _log_dict(self, metrics, **kwargs):
        default_args = {
            "on_step": True,
            "on_epoch": True,
            "prog_bar": True,
            "add_dataloader_idx": False,
            "sync_dist": True,
        }
        default_args.update(kwargs)
        self.log_dict(metrics, **default_args)

    def step(self, batch, prefix="train", on_step=False):
        input, targets, dynamic_kwargs = batch

        output = self.model(input)
        loss, metrics = self.trainer.datamodule.loss(
            input=input,
            output=output,
            targets=targets,
            prefix=prefix,
            **dynamic_kwargs,
        )
        self._log_dict(metrics, on_step=on_step)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, on_step=True)
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self._log_dict(loss_epoch, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.step(batch, prefix="val")

    def get_scheduler(self, optimizer):
        lr_scheduler_cls, scheduler_hparams = get_object_from_registry(
            self.hparams.scheduler, scheduler_registry, init=False
        )
        interval = scheduler_hparams.pop("interval", "epoch")
        lr_scheduler = lr_scheduler_cls(optimizer, **scheduler_hparams)
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": interval,  # 'epoch' or 'step'
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        return scheduler

    def configure_optimizers(self):
        # get param groups
        params_dict = defaultdict(list)
        for p in self.parameters():

            def get_key(x):
                return "normal" if not x else frozenset(x.items())

            params_dict[get_key(getattr(p, "_optim", None))].append(p)

        # add param groups to optimizer
        optimizer = torch.optim.Adam(
            params_dict.pop("normal"), **self.hparams.optimizer
        )

        hp_list = [dict(hp) for hp in params_dict]
        print("Hyperparameter groups", hp_list)
        for hp, hp_group in params_dict.items():
            optimizer.add_param_group(
                {"params": hp_group, **self.hparams.optimizer, **dict(hp)}
            )

        # Print optimizer info for debugging
        unique_hparams = {k for hp in hp_list for k in hp}
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in unique_hparams}
            log.info(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )

        if self.hparams.scheduler is None:
            return optimizer
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

        return [optimizer], [lr_scheduler]
