import tqdm
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from common.metrics import *
from models.interaction_models import Concat, Dot

class InteractionTrainer():
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.categorical_ids = categorical_ids
        self.device = device

        self.num_users = len(self.categorical_ids['user'])
        self.num_items = len(self.categorical_ids['champion'])

        self.model = self._initialize_model()
        self.criterion = self._initialize_criterion()
        self.optimizer = self._initialize_optimizer()

    def _initialize_model(self):
        if self.args.model_type == 'concat':
            model = Concat(self.args, self.categorical_ids)
        elif self.args.model_type == 'dot':
            model = Dot(self.args, self.categorical_ids)
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

    def _save_model(self, model_name, epoch):
        model_path = wandb.run.dir + '/' + str(model_name)
        torch.save({"state_dict": self.model.state_dict(),
                    "epoch": epoch}, model_path)

    def train(self):
        args = self.args
        # evaluate the initial-run
        print('[Epoch:0]')
        summary = self.evaluate(self.val_loader)
        wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            losses = []
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = y_batch.to(self.device).unsqueeze(1)
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                summary['loss'] = np.mean(losses)
                wandb.log(summary, e)
                self._save_model('model.pt', e)

    def evaluate(self, loader):
        self.model.eval()
        # initialize metrics
        summary = {}
        for k in self.args.k_list:
            summary['HR@'+str(k)] = 0.0
            summary['NDCG@'+str(k)] = 0.0
        summary['MRR'] = 0.0

        # measure metrics
        UNK = 0
        num_evaluation = 0
        for _, y_batch in tqdm.tqdm(loader):
            user_batch, item_batch, _ = y_batch
            for idx in range(len(user_batch)):
                user = int(user_batch[idx].item())
                # only test for the valid user in the training-set
                if user != UNK:
                    item = int(item_batch[idx].item())
                    single_summary = self._evaluate_single(user, item)
                    for metric, value in single_summary.items():
                        summary[metric] += value
                    num_evaluation += 1
        for metric, value in summary.items():
            summary[metric] = (value / num_evaluation) * 100
        return summary

    def _evaluate_single(self, user, item):
        """
        :param user: (int) index of the user
        :param item: (int) index of the corresponding item
        :return:
        """
        user_batch = torch.LongTensor([user] * self.num_items).to(self.device)
        item_batch = torch.arange(self.num_items).to(self.device)
        x = (user_batch, item_batch)
        logit = self.model(x).reshape(1, -1)
        y_true = torch.eye(self.num_items)[item].unsqueeze(0).numpy()
        y_pred = F.sigmoid(logit).detach().cpu().numpy()

        summary = {}
        for k in self.args.k_list:
            summary['HR@'+str(k)] = recall_at_k(y_true, y_pred, k)
            summary['NDCG@'+str(k)] = ndcg_at_k(y_true, y_pred, k)
        summary['MRR'] = average_precision_at_k(y_true, y_pred, y_true.shape[1])

        return summary
