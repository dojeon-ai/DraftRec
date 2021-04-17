import tqdm
import wandb
import torch
import torch.nn.functional as F
from common.metrics import *
from models.user_rec_models import UserRec, SPOP
from trainers.base_trainer import BaseTrainer


class UserRecTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        super(UserRecTrainer, self).__init__(args, train_loader, val_loader, test_loader, categorical_ids, device)
        self.model = self._initialize_model()
        self.pi_criterion, self.v_criterion = self._initialize_criterion()
        self.optimizer, self.scheduler = self._initialize_optimizer()

    def _initialize_model(self):
        if self.args.model_type in ['sas', 'bert']:
            model = UserRec(self.args, self.categorical_ids, self.device)
        elif self.args.model_type == 'spop':
            model = SPOP(self.args, self.categorical_ids, self.device)
        else:
            raise NotImplementedError
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
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                if self.args.model_type == 'spop':
                    continue
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = [feature.to(self.device) for feature in y_batch]
                (v_true, v_mask, pi_true) = y_batch

                pi_pred, v_pred = self.model(x_batch)
                N, S, C = pi_pred.shape
                pi_loss = self.pi_criterion(pi_pred.reshape(N*S, C), pi_true.reshape(-1))
                v_loss = self.v_criterion(v_pred[v_mask == 1].squeeze(-1), v_true[v_mask == 1])
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

    #def evaluate(self, loader):
        #total_num_valid_users = 0
        #for x_batch, y_batch in tqdm.tqdm(loader):
        #    x_batch = [feature.to(self.device) for feature in x_batch]
        #    y_batch = [feature.to(self.device) for feature in y_batch]
        #    (team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids) = x_batch
        #    (win_labels, item_labels) = y_batch  # [N,S], [N,S]
        #
        #    item_labels, item_labels_idx = torch.max(item_labels, 1)  # [N]
        #    item_labels = torch.eye(self.num_items)[item_labels].detach().cpu().numpy()  # [N, C]
        #    UNK = 3
        #    user_x_batch, user_y_batch = [], []
        #    for batch_idx, seq_idx in enumerate(item_labels_idx):
        #        user_idx = user_ids[batch_idx][seq_idx].item()
        #        history_idx = history_ids[batch_idx][seq_idx].item()
        #        if user_idx != UNK:
        #            user_x, user_y = self.train_loader.dataset.get_item_with_history_idx(user_idx, history_idx)
        #            user_x_batch.append(user_x)
        #            user_y_batch.append(user_y)
        #
        #    user_x_batch = [torch.stack(feature).to(self.device) for feature in list(map(list, zip(*user_x_batch)))]
        #    user_y_batch = [torch.stack(feature).to(self.device) for feature in list(map(list, zip(*user_y_batch)))]
        #    (_, _, user_item_labels) = user_y_batch
        #    pi, v = self.model(user_x_batch)
        #    pi = torch.exp(pi)
        #    v = F.sigmoid(v)
        #    # select the right sequence (pi: [last])
        #    N, S, C = pi.shape
        #    pi = pi[torch.arange(N, device=self.device), -1, :].detach().cpu().numpy()
        #    user_item_labels = user_item_labels[torch.arange(N, device=self.device), -1].detach().cpu().numpy()
        #    user_item_labels = np.eye(self.num_items)[user_item_labels]
