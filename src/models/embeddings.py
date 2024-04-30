import torch.nn as nn


class SimpleCnaEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
    ):
        super().__init__()

        self.value_embeddings = nn.Linear(input_dim, embed_dim)

    def forward(self, inputs):
        return self.value_embeddings(inputs)


registry = {
    "simple_cna": SimpleCnaEmbedding,
}
