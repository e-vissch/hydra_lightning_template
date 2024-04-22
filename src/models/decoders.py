import torch.nn as nn


class SimpleDecoder(nn.Module):
    def __init__(self, decoder_dim, out_dim):
        super().__init__()

        self.decoder = nn.Linear(decoder_dim, out_dim)

    def forward(self, decoder_inputs):
        return self.decoder(decoder_inputs)


registry = {
    "simple": SimpleDecoder,
}
