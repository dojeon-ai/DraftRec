import torch
import torch.nn as nn
import numpy as np
import math


def init_transformer_weights(module, init_range=0.0625):
    """ Initialize the weights """
    if isinstance(module, (nn.Embedding, nn.Linear)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=init_range)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    return module
