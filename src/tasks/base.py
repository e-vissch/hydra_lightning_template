from dataclasses import dataclass

from torch.nn import functional as F


@dataclass
class BaseTask:
    def default_loss(self, input, output):
        return F.mse_loss(input, output)

    def process_for_loss(self, **kwargs):
        return kwargs


task_registry = {"base": BaseTask}
