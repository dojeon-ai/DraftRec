import torch
import tqdm
import random
import numpy as np
from dataset.rec_eval_dataset import RecEvalDataset


class DraftRecDataset(RecEvalDataset):
    def __init__(self, args, match_data, user_history_data, categorical_ids):
        super(DraftRecDataset, self).__init__(args, match_data, user_history_data, categorical_ids)
