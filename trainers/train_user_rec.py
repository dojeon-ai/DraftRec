import tqdm
import wandb
import torch
import torch.nn.functional as F
from common.metrics import *
from models.user_rec_models import UserRec

class UserRecTrainer():
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
        self.policy_criterion, self.value_criterion = self._initialize_criterion()
        self.optimizer, self.scheduler = self._initialize_optimizer()

    def _initialize_model(self):
        model = UserRec(self.args, self.categorical_ids, self.device)
        return model.to(self.device)

    def _initialize_criterion(self):
        policy_criterion = torch.nn.NLLLoss(ignore_index=0).to(self.device)
        value_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        return policy_criterion, value_criterion

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.args.lr,
                                                        epochs=self.args.epochs,
                                                        steps_per_epoch=len(self.train_loader),
                                                        pct_start=0.2)
        return optimizer, scheduler

    def _save_model(self, model_name, epoch):
        model_path = wandb.run.dir + '/' + str(model_name)
        torch.save({"state_dict": self.model.state_dict(),
                    "epoch": epoch}, model_path)

    def train(self):
        args = self.args
        policy_losses = []
        value_losses = []

        # evaluate the initial-run
        summary = {}
        #summary = self.evaluate(self.val_loader)
        best_HR1 = 0
        best_epoch = 0
        #wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = [feature.to(self.device) for feature in y_batch]
                (win_labels, win_mask_labels, item_labels) = y_batch

                pi, v = self.model(x_batch)
                N, S, C = pi.shape
                policy_loss = self.policy_criterion(pi.reshape(N*S, C), item_labels.reshape(-1))
                value_loss = self.value_criterion(v[win_mask_labels==1].squeeze(-1), win_labels[win_mask_labels==1])
                loss = (1-args.lmbda) * policy_loss + args.lmbda * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                # self.scheduler.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            if e % args.evaluate_every == 0:
                #summary = self.evaluate(self.val_loader)
                #if summary['HR@1'] > best_HR1:
                    #best_epoch = e
                    #best_HR1 = summary['HR@1']
                summary['best_epoch'] = best_epoch
                summary['policy_loss'] = np.mean(policy_losses)
                summary['value_loss'] = np.mean(value_losses)
                wandb.log(summary, e)
                # self._save_model('model.pt', e)

    def evaluate(self, loader):
        pass
