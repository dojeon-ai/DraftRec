from .base import BaseTrainer
from ..common.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MatchTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, model):
        super().__init__(args, train_loader, val_loader, test_loader, model)
        
    @classmethod
    def code(cls):
        return 'match'

    def calculate_loss(self, batch):
        loss, extra_metrics = None, {}
        pi, v = self.model(batch)
        
        pi_loss = F.cross_entropy(pi, batch['target_champion'].flatten(), ignore_index=self.args.PAD)
        v_loss = F.binary_cross_entropy(F.sigmoid(v), batch['outcome'])
        
        lmbda = self.args.lmbda
        loss = lmbda * pi_loss + (1-lmbda) * v_loss
        cnt = len(pi)
        extra_metrics['pi_loss'] = (pi_loss.item(), cnt)
        extra_metrics['v_loss'] = (v_loss.item(), cnt)
        
        return loss, extra_metrics

    def calculate_metrics(self, batch):
        args = self.args
        metrics = {}
        pi, v = self.model(batch)
        pi = pi.cpu().numpy()
        v = F.sigmoid(v).cpu().numpy().flatten()
        
        pi_true = batch['target_champion'].cpu().numpy()
        pi_true = np.eye(args.num_champions)[pi_true].squeeze(1)
        v_true = batch['outcome'].cpu().numpy().flatten()
        is_draft_finished = batch['is_draft_finished'].cpu().numpy().flatten()
        
        pi = pi[~is_draft_finished]
        pi_true = pi_true[~is_draft_finished]
        
        rec_metrics = get_recommendation_metrics_for_ks(pi, pi_true, args.metric_ks)
        win_metrics = get_win_prediction_metrics(v, v_true, is_draft_finished)
        metrics.update(rec_metrics)
        metrics.update(win_metrics)
        
        return metrics
