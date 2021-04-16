import tqdm
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from common.metrics import *
from models.interaction_models import NMF, DMF
from trainers.base_trainer import BaseTrainer


class InteractionTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        super(InteractionTrainer, self).__init__(args, train_loader, val_loader, test_loader, categorical_ids, device)
        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()
        self.optimizer = self._initialize_optimizer()

    def _initialize_model(self):
        if self.args.model_type == 'nmf':
            model = NMF(self.args, self.categorical_ids)
        elif self.args.model_type == 'dmf':
            model = DMF(self.args, self.categorical_ids)
        else:
            raise NotImplementedError
        return model.to(self.device)

    def _initialize_criterion(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        return criterion.to(self.device)

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr)
        return optimizer

    def train(self):
        args = self.args
        losses = []
        # evaluate the initial-run
        summary = self.evaluate(self.val_loader)
        best_HR1 = 0
        best_HR1_epoch = 0
        wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = [feature.to(self.device) for feature in y_batch]
                logits = self.model(x_batch).squeeze(1)

                # binarized target for nmf
                if self.args.model_type == 'nmf':
                    targets = torch.sign(y_batch[0].float())
                # explicit probabilistic target for dmf
                elif self.args.model_type == 'dmf':
                    eps = 1e-5
                    targets = (y_batch[0].float() / (y_batch[1] + eps))
                else:
                    raise NotImplementedError

                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                if summary['HR@1'] > best_HR1:
                    best_HR1_epoch = e
                    best_HR1 = summary['HR@1']
                summary['best_HR@1_epoch'] = best_HR1_epoch
                summary['loss'] = np.mean(losses)
                wandb.log(summary, e)
                self._save_model('model.pt', e)
                losses = []