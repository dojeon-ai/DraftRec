from .base import BaseTrainer
import torch
import torch.nn as nn
import numpy as np


class SequentialTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, model):
        super().__init__(args, train_loader, val_loader, test_loader, model)
        
    @classmethod
    def code(cls):
        return 'sequential'

    def calculate_loss(self, batch):
        loss, extra_info = None, {}
        
        return loss, extra_info

    def calculate_metrics(self, batch):
        metrics = {}
        
        return metrics