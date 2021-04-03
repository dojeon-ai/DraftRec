import tqdm
import wandb
import torch
import torch.nn.functional as F
from common.metrics import *


class UserRecTrainer():
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.categorical_ids = categorical_ids
        self.device = device
        pass

    def _initialize_model(self):
        pass

    def _initialize_criterion(self):
        pass

    def _initialize_optimizer(self):
        pass

    def _save_model(self, model_name, epoch):
        pass

    def train(self):
        args = self.args
        # evaluate the initial-run
        print('[Epoch:0]')
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            losses = []
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                (ban_ids, user_ids, item_ids, lane_ids, win_ids) = x_batch
                (win_labels, win_mask_labels, item_labels) = y_batch  # [N,S], [N,S]
                import pdb
                pdb.set_trace()

    def evaluate(self, loader):
        pass