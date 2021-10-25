from .base import BaseTrainer
import torch
import torch.nn as nn
import numpy as np


class DraftRecTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, model):
        super().__init__(args, train_loader, val_loader, test_loader, model)
        
    @classmethod
    def code(cls):
        return 'draftrec'

    def calculate_loss(self, batch):
        loss, extra_info = None, {}
        import pdb
        pdb.set_trace()
        """
        logits = self.model(batch['items'], batch['ratings'], batch['candidates'])
        B, T, C = logits.shape
        
        logits = logits.view(B*T, -1)
        labels = batch['labels'].view(B*T)
        masks = batch['masks'].view(B*T)
        loss = self.criterion(logits, labels)
        loss = ((1-masks) * loss).mean()
        """
        
        return loss

    def calculate_metrics(self, batch):
        metrics = {}
        import pdb
        pdb.set_trace()
        """
        logits = self.model(batch['items'], batch['ratings'], batch['candidates'])
        B, T, C = logits.shape
        
        logits = logits.view(B*T, -1)
        labels = batch['labels'].view(B*T)
        masks = batch['masks'].view(B*T)
        
        logits = torch.masked_select(logits, (masks==0).unsqueeze(1)).view(-1, C)
        labels = torch.masked_select(labels, masks==0)
        
        metrics = recalls_and_ndcgs_for_ks(logits.cpu().numpy(), labels.cpu().numpy(), self.metric_ks)
        """
        return metrics
