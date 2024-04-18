

from functools import partial
import torch 
import torch.nn as nn
from configs.schemas import ModelConfig
from src.utils.config import get_object_from_registry

embedding_registry = {}
backbone_registry = {}


class ExampleModel(nn.Module):
    def __init__(self, config: ModelConfig, decoder):
        super().__init__()

        self.embeddings = get_object_from_registry(
            config.embeddings, embedding_registry
        )

        self.backbone = get_object_from_registry(config.backbone, backbone_registry)

        # decoder depends on task
        self.decoder = decoder


    def hidden_embeddings(self, inputs, position_features=None):
        input_embeddings = self.embeddings(inputs, position_features=position_features)
        return self.backbone(input_embeddings)

    def forward(self, inputs, position_features=None):
        hidden_states = self.hidden_embeddings(
            inputs, position_features=position_features
        )
        return self.decoder(hidden_states)
    

    def inference(self, inputs, position_features=None, hidden=False):
        self.eval()
        with torch.inference_mode():
            infer_func = self.hidden_embeddings if hidden else partial(self.forward)
            output = infer_func(inputs, position_features=position_features)
        self.train()
        return output