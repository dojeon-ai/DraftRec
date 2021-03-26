import tqdm
import wandb
import torch
import torch.nn.functional as F
from common.metrics import *
from models.recommendation_models import Transformer


class RecommendationModelTrainer():
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
        model = Transformer(self.args, self.categorical_ids)
        return model.to(self.device)

    def _initialize_criterion(self):
        pass

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr)
        return optimizer

    def train(self):
        args = self.args
        losses = []
        # evaluate the initial-run
        #summary = self.evaluate(self.val_loader)
        #wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                #y_batch = y_batch.to(self.device).unsqueeze(1)

                logits = self.model(x_batch)
                #loss = self.criterion(logits, y_batch)
                #loss.backward()
                #losses.append(loss.item())
                #self.optimizer.step()

            #if e % args.evaluate_every == 0:
            #    summary = self.evaluate(self.val_loader, e)
            #    wandb.log(summary, e)


    def evaluate(self, loader):
        pass
