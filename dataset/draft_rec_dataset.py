import torch
import tqdm
import random
import numpy as np
from dataset.eval_dataset import EvalDataset


class DraftRecDataset(EvalDataset):
    def __init__(self, args, match_data, user_history_data, categorical_ids):
        super(DraftRecDataset, self).__init__(args, match_data, user_history_data, categorical_ids)
