from functools import partial
from typing import Dict, List

import torch.nn as nn

from src.models import standalone_hyenadna as hyena

try:
    from mamba_ssm.models import mixer_seq_simple as mamba
except ImportError:
    # if on cpu/local
    mamba = None


class HyenaBackbone(nn.Module):
    # adapted from LMBackbone Hyena
    # doesn't necessarily use HyenaOperator, can use just MHA/linear attn
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        layer=None,
        initializer_cfg=None,
        resid_dropout: float = 0.0,
        attn_layer_idx: List[int] | None = None,
        attn_cfg: Dict | None = None,
    ) -> None:
        """
        Follows design from Hyena. If attn_layer_idx = [0, 1] n_layer = 2,
        then both layers will be MHA not HyenaOperator.
        """

        super().__init__()
        print("Using Normal Backbone")

        self.layers = nn.ModuleList(
            [
                hyena.create_block(
                    d_model,
                    d_inner=d_inner,
                    layer=layer,
                    layer_idx=i,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                )
                for i in range(n_layer)
            ]
        )

        self.drop_f = nn.Dropout(resid_dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.apply(
            partial(
                hyena._init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, embeddings):
        residual = None
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))


class MambaBackbone(nn.Module):
    # almost identical to Mamba MixerModel, remove fused stuff for now.
    # Don't want to create a shared class as they may diverge in the future.
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        residual_in_fp32=True,
    ) -> None:
        super().__init__()
        print("Using MambaBackbone")
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.

        self.layers = nn.ModuleList(
            [
                mamba.create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)

        self.apply(
            partial(
                mamba._init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, embeddings, inference_params=None):
        hidden_states = embeddings
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states


registry = {"attn": HyenaBackbone, "hyena": HyenaBackbone, "mamba": MambaBackbone}
