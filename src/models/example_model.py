from functools import partial
import torch 
import torch.nn as nn
from configs.schemas import ModelConfig
from src.models.standalone_hyenadna import _init_weights
from src.utils.config import get_object_from_registry

from src.models.embeddings import registry as embedding_registry
from src.models.backbone import registry as backbone_registry



class ExampleModel(nn.Module):
    def __init__(self, config: ModelConfig, decoder):
        super().__init__()

        self.embeddings = get_object_from_registry(
            config.embeddings, embedding_registry
        )

        self.backbone = get_object_from_registry(config.backbone, backbone_registry)

        # decoder depends on task
        self.decoder = decoder

        # Initialize weights 
        self.apply(partial(_init_weights, n_layer=config.n_layer))


    def forward(self, inputs):
        input_embeddings = self.embeddings(inputs)
        hidden_states = self.hidden_embeddings(input_embeddings)
        return self.decoder(hidden_states)
    

    def inference(self, inputs):
        self.eval()
        with torch.inference_mode():
            output = self.forward(inputs)
        self.train()
        return output