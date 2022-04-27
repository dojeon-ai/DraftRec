from ..common.logger import LoggerService, AverageMeterSet
from ..common.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from abc import *
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import copy


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, train_loader, val_loader, test_loader, model):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = args.device
        self.model = model.to(self.device)
        self.use_parallel = args.use_parallel
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)
        
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        self.clip_grad_norm = args.clip_grad_norm

        self.logger = LoggerService(args)
        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        
        self.best_metric_value = 0
        self.best_epoch = -1
        self.steps = 0

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), 
                             lr=args.lr, weight_decay=args.weight_decay, 
                             momentum=args.momentum)
        else:
            raise ValueError
            
    def _create_scheduler(self):
        args = self.args
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
    
    def _create_state_dict(self, epoch):
        return {
            'model_state_dict': self.model.module.state_dict() if self.use_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': epoch
        }
            
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        """
        metrics: {key: (value, cnt)}
        """
        pass

    def train(self):
        # validation at an initialization
        val_log_data = self.validate(mode='val')
        val_log_data['epoch'] = 0
        self.logger.log_val(val_log_data)
        
        for epoch in range(1, self.num_epochs+1):
            # train
            train_log_data = self.train_one_epoch()
            train_log_data['epoch'] = epoch
            self.logger.log_train(train_log_data)
            
            # validation
            val_log_data = self.validate(mode='val')
            val_log_data['epoch'] = epoch
            self.logger.log_val(val_log_data)            

            # update the best_model
            cur_metric_value = val_log_data[self.best_metric]
            if cur_metric_value > self.best_metric_value:
                self.best_metric_value = cur_metric_value
                self.best_epoch = epoch
                best_model_state_dict = self._create_state_dict(epoch)
                self.logger.save_state_dict(best_model_state_dict)
             
            self.lr_scheduler.step()
        
        # test with the best_model
        best_model_state = self.logger.load_state_dict()['model_state_dict']
        if self.use_parallel:
            self.model.module.load_state_dict(best_model_state)
        else:
            self.model.load_state_dict(best_model_state)
        test_log_data = self.validate(mode='test')
        self.logger.log_test(test_log_data)

    def train_one_epoch(self):
        average_meter_set = AverageMeterSet()
        self.model.train()
        for batch in tqdm(self.train_loader):

            batch_size = next(iter(batch.values())).size(0)
            # Need to send to cuda before forwarding in the model.
            batch = {k:v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            loss, extra_metrics = self.calculate_loss(batch)
            average_meter_set.update('loss', loss.item())
            for k, v in extra_metrics.items():
                average_meter_set.update(k, v[0], n=v[1])
            
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.steps += 1
            
            # log the training information from the training process
            log_data = {'step': self.steps}
            log_data.update(average_meter_set.averages())
            self.logger.log_train(log_data)
            
        log_data = {'step': self.steps}
        log_data.update(average_meter_set.averages())
        
        return log_data

    def validate(self, mode):
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise ValueError

        average_meter_set = AverageMeterSet()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k:v.to(self.device) for k, v in batch.items()}
                metrics = self.calculate_metrics(batch)
                
                for k, v in metrics.items():
                    average_meter_set.update(k, v[0], n=v[1])

        log_data = {'step': self.steps}
        log_data.update(average_meter_set.averages())

        return log_data
