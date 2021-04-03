import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import EncoderLayer
from models.modules import GELU, PositionalEncoding, LayerNorm


class UserRec(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(UserRec, self).__init__()
        pass