from dataclasses import dataclass

from src.utils.metrics import mse


@dataclass
class BaseTask:
    # Designed for subclassing. 
    # Define custom logic for loss/metrics/data processing for task
    decoder_cls = ...

    @property
    def decoder_cls(self):
        return self._decoder_cls

    def default_loss(self, _input, output, targets, sample_len):
        return mse(output, targets, sample_len)

    def process_for_loss(self, **kwargs):
        return kwargs

    def remove_extra(self, *args, prepend_len=0, len_no_postpend=None):
        return tuple(arg[:, prepend_len:len_no_postpend] for arg in args)


task_registry = {"base": BaseTask}