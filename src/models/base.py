import torch.nn as nn
from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def load(self, model_state, use_parallel):
        if use_parallel:
            self.module.load_state_dict(model_state)
        else:
            self.load_state_dict(model_state)
