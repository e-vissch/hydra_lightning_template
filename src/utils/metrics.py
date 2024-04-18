import torch
import torch.nn.functional as F


def get_output_mask(outs, sample_len):
    mask = torch.zeros_like(outs, dtype=torch.bool)
    for i, length in enumerate(sample_len):
        mask[i, :length] = 1
    return mask


def get_nonzero_batch_vals(outs, y, sample_len):
    # Computes the loss of the first `lens` items in the batches
    mask = get_output_mask(outs, sample_len)
    outs_masked = torch.masked_select(outs, mask)
    y_masked = torch.masked_select(y, mask)
    return outs_masked, y_masked



def mse(outs, y, sample_len=None, reduction="mean"):
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if sample_len is None:
        return F.mse_loss(outs, y)
    return F.mse_loss(*get_nonzero_batch_vals(outs, y, sample_len), reduction=reduction)



output_metric_fns = {
    "mse": mse,
}
