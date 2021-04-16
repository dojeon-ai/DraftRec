import tqdm
import wandb
import torch
import torch.nn.functional as F
from common.metrics import *
from common.plot_utils import plot_attn_weights
from models.context_rec_models import ContextRec
from trainers.base_trainer import BaseTrainer


class RewardModelTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        super(RewardModelTrainer, self).__init__(args, train_loader, val_loader, test_loader, categorical_ids, device)
        self.model = self._initialize_model()
        self.v_criterion = self._initialize_criterion()
        self.optimizer, self.scheduler = self._initialize_optimizer()

    def _initialize_model(self):
        model = ContextRec(self.args, self.categorical_ids, self.device)
        return model.to(self.device)

    def _initialize_criterion(self):
        v_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        return v_criterion

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self. args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.args.lr,
                                                        epochs=self.args.epochs,
                                                        steps_per_epoch=len(self.train_loader),
                                                        pct_start=0.2)
        return optimizer, scheduler

    def train(self):
        args = self.args
        v_losses = []
        # evaluate the initial-run
        summary = self.evaluate(self.val_loader)
        best_acc = 0
        best_epoch = 0
        wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = [feature.to(self.device) for feature in y_batch]
                (v_true, _) = y_batch
                _, v_pred, attn = self.model(x_batch, return_attn=True)
                attn = attn.detach().cpu().numpy()

                #v_loss = self.v_criterion(v_pred.squeeze(-1), v_true)
                v_loss = self.v_criterion(v_pred[:,0,:].squeeze(-1), v_true[:,0])
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                # self.scheduler.step()
                v_losses.append(v_loss.item())

            if e % args.evaluate_every == 0:
                plot_attn_weights(attn)
                summary = self.evaluate(self.val_loader)
                summary['v_loss'] = np.mean(v_losses)
                wandb.log(summary, e)
                # self._save_model('model.pt', e)
                v_losses = []
