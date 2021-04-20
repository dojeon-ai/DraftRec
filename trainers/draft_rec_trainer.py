import tqdm
import wandb
import torch
import torch.nn.functional as F
from common.metrics import *
from models.draft_rec_models import DraftRec
from trainers.base_trainer import BaseTrainer


class DraftRecTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        super(DraftRecTrainer, self).__init__(args, train_loader, val_loader, test_loader, categorical_ids, device)
        self.model = self._initialize_model()
        self.pi_criterion, self.v_criterion = self._initialize_criterion()
        self.optimizer, self.scheduler = self._initialize_optimizer()

    def _initialize_model(self):
        model = DraftRec(self.args, self.categorical_ids, self.device)
        return model.to(self.device)

    def _initialize_criterion(self):
        pi_criterion = torch.nn.NLLLoss(ignore_index=0).to(self.device)
        v_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        return pi_criterion, v_criterion

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

    def train(self):
        args = self.args
        pi_losses = []
        v_losses = []
        # evaluate the initial-run
        summary = self.evaluate(self.val_loader)
        best_HR1 = 0
        best_epoch = 0
        wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for match_batch, user_history_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                _, match_y_batch = match_batch
                match_y_batch = [feature.to(self.device) for feature in match_y_batch]
                user_history_x_batch, user_history_y_batch = user_history_batch
                user_history_x_batch = [feature.to(self.device) for feature in user_history_x_batch]
                user_history_y_batch = [feature.to(self.device) for feature in user_history_y_batch]
                (ban_ids, item_ids, lane_ids, win_ids) = user_history_x_batch
                (win_labels, win_mask_labels, item_labels) = user_history_y_batch
                (v_true, pi_true, _) = match_y_batch
                N, _ = pi_true.shape

                turn = torch.argmax(pi_true, 1)-1
                next_item_ids = item_ids.clone()
                next_item_ids[torch.arange(N, device=self.device), turn, -1] = \
                    (item_labels[torch.arange(N, device=self.device), turn, -1])
                next_user_history_x_batch = (ban_ids, next_item_ids, lane_ids, win_ids)

                # forward & backward
                pi_pred, _ = self.model(user_history_x_batch)
                _, v_pred = self.model(next_user_history_x_batch)
                _, _, C = pi_pred.shape
                pi_loss = self.pi_criterion(pi_pred.reshape(-1, C), pi_true.reshape(-1))
                v_loss = self.v_criterion(v_pred[:,0,:].squeeze(-1), v_true[:,0])
                loss = (1-args.lmbda) * pi_loss + args.lmbda * v_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                self.scheduler.step()
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                if summary['HR@1'] > best_HR1:
                    best_epoch = e
                    best_HR1 = summary['HR@1']
                summary['best_epoch'] = best_epoch
                summary['pi_loss'] = np.mean(pi_losses)
                summary['v_loss'] = np.mean(v_losses)
                wandb.log(summary, e)
                # self._save_model('model.pt', e)
                pi_losses = []
                v_losses = []
