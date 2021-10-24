from .base import BaseModel
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseModel)

MODELS = {c.code():c
          for c in all_subclasses(BaseModel)
          if c.code() is not None}


def init_model(args):
    model = MODELS[args.model_type]
    return model(args)
