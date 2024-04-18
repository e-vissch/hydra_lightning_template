import abc
from functools import cached_property, partial

import lightning as L

from src.tasks.base import BaseTask
from src.utils.metrics import output_metric_fns


class BaseDataModule(L.LightningDataModule, abc.ABC):
    def __init__(self, config, task: BaseTask):
        super().__init__()

        self.loss_name = config.loss
        self.loss_kwargs = config.loss_kwargs
        self.config_metrics = config.metrics

        self.task = task  # type: ignore

    @cached_property
    def loss_func(self):
        loss_func = getattr(
            self.task, self.loss_name, output_metric_fns.get(self.loss_name)
        )
        if loss_func is None:
            raise ValueError(
                f"Loss function {self.loss_name} not found in task or output_metric_fns"
            )
        return partial(loss_func, **self.loss_kwargs)

    def loss(self, prefix="train", **kwargs):
        kwargs = self.task.process_for_loss(**kwargs)
        loss = self.loss_func(**kwargs)
        metrics = self.metrics(prefix=prefix, **kwargs)
        return loss, metrics | {f"{prefix}/loss": loss}

    @cached_property
    def metric_names(self):
        return {
            metric: getattr(
                self.task,
                metric,
                getattr(self, metric, output_metric_fns.get(metric)),
            )
            for metric in self.config_metrics
        }

    def metrics(self, *args, prefix=None, **kwargs):
        print_prefix = "" if prefix is None else f"{prefix}/"
        metric_dict = {}
        for metric_name, metric_fn in self.metric_names.items():
            metric_result = metric_fn(*args, **kwargs, prefix=prefix)
            if isinstance(metric_result, dict):
                metric_dict.update(metric_result)
            else:
                metric_dict[f"{print_prefix}{metric_name}"] = metric_result

        return metric_dict
