import tqdm
import wandb
import copy
import torch
import torch.nn.functional as F
from common.metrics import *
from models.draft_rec_models import DraftRec
from trainers.base_trainer import BaseTrainer
from torch.distributions import Categorical


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
        best_ACC = 0
        best_epoch = 0
        wandb.log(summary, 0)
        grad_accum_step = (self.args.batch_size // self.args.data_batch_size)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for step, (match_batch, user_history_batch) in tqdm.tqdm(enumerate(self.train_loader)):
                _, match_y_batch = match_batch
                match_y_batch = [feature.to(self.device).squeeze(0) for feature in match_y_batch]
                user_history_x_batch, user_history_y_batch = user_history_batch
                user_history_x_batch = [feature.to(self.device).squeeze(0) for feature in user_history_x_batch]
                user_history_y_batch = [feature.to(self.device).squeeze(0) for feature in user_history_y_batch]

                (ban_ids, item_ids, lane_ids, stat_ids, win_ids) = user_history_x_batch
                (win_labels, win_mask_labels, item_labels) = user_history_y_batch
                (v_true, pi_true, _) = match_y_batch
                N, _ = pi_true.shape

                turn = torch.argmax(pi_true, 1)-1
                next_item_ids = item_ids.clone()
                next_item_ids[torch.arange(N, device=self.device), turn, -1] = \
                    (item_labels[torch.arange(N, device=self.device), turn, -1])
                next_user_history_x_batch = (ban_ids, next_item_ids, lane_ids, stat_ids, win_ids)

                # forward & backward
                pi_pred, _ = self.model(user_history_x_batch)
                _, v_pred = self.model(next_user_history_x_batch)
                _, _, C = pi_pred.shape
                pi_loss = self.pi_criterion(pi_pred.reshape(-1, C), pi_true.reshape(-1))
                v_loss = self.v_criterion(v_pred[:,0,:].squeeze(-1), v_true[:,0])
                loss = (1-args.lmbda) * pi_loss + args.lmbda * v_loss
                loss = loss / grad_accum_step
                loss.backward()
                if step % grad_accum_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.scheduler.step()
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                summary['best_epoch'] = best_epoch
                summary['pi_loss'] = np.mean(pi_losses)
                summary['v_loss'] = np.mean(v_losses)
                wandb.log(summary, e)
                if summary['ACC'] > best_ACC:
                    best_epoch = e
                    best_ACC = summary['ACC']
                    summary = self.evaluate(self.test_loader, prefix='TEST_')
                    wandb.log(summary, e)
                    self._save_model('best.pt', e)
                pi_losses = []
                v_losses = []

        summary = self.evaluate(self.test_loader, prefix='TEST_')
        wandb.log(summary, e)
        self._save_model('last.pt', e)

    def finetune(self):
        try:
            self.model.load_state_dict(torch.load(self.args.pretrained_path)['state_dict'])
        except:
            raise NotImplementedError
        actor = self.model
        critic = copy.deepcopy(self.model)
        args = self.args
        sl_losses = []
        rl_losses = []
        # evaluate the initial-run

        #summary = self.evaluate(self.val_loader)
        self._evaluate_rew(self.val_loader, actor, critic)

        best_ACC = 0
        best_epoch = 0
        #wandb.log(summary, 0)
        grad_accum_step = (self.args.batch_size // self.args.data_batch_size)
        # start training
        for e in range(1, args.epochs + 1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for step, (match_batch, user_history_batch) in tqdm.tqdm(enumerate(self.train_loader)):
                _, match_y_batch = match_batch
                match_y_batch = [feature.to(self.device).squeeze(0) for feature in match_y_batch]
                user_history_x_batch, user_history_y_batch = user_history_batch
                user_history_x_batch = [feature.to(self.device).squeeze(0) for feature in user_history_x_batch]
                user_history_y_batch = [feature.to(self.device).squeeze(0) for feature in user_history_y_batch]

                (ban_ids, item_ids, lane_ids, stat_ids, win_ids) = user_history_x_batch
                (win_labels, win_mask_labels, item_labels) = user_history_y_batch
                (_, pi_true, _) = match_y_batch
                N, _ = pi_true.shape
                turn = torch.argmax(pi_true, 1) - 1

                # forward actor
                pi_pred, _ = actor(user_history_x_batch)
                _, _, C = pi_pred.shape
                dist = Categorical(logits=pi_pred[torch.arange(N, device=self.device), turn+1, :])
                action = dist.sample()
                next_item_ids = item_ids.clone()
                next_item_ids[torch.arange(N, device=self.device), turn, -1] = action
                next_user_history_x_batch = (ban_ids, next_item_ids, lane_ids, stat_ids, win_ids)
                # forward critic
                adv = torch.zeros(N, device=self.device)
                with torch.no_grad():
                    _, v_pred = critic(user_history_x_batch)
                    _, q_pred = critic(next_user_history_x_batch)
                    v_pred = F.sigmoid(v_pred[:, 0, :])
                    q_pred = F.sigmoid(q_pred[:, 0, :])
                    for i in range(N):
                        if turn[i] in [0, 2, 3, 6, 7]:
                            adv[i] = q_pred[i] - v_pred[i]
                        else:
                            adv[i] = (1-q_pred[i]) - (1-v_pred[i])
                # compute loss
                log_prob_act = dist.log_prob(action)
                rl_loss = -(log_prob_act * adv).mean()
                sl_loss = self.pi_criterion(pi_pred.reshape(-1, C), pi_true.reshape(-1))
                loss = (1 - args.lmbda_rl) * sl_loss + args.lmbda_rl * rl_loss
                loss = loss / grad_accum_step
                loss.backward()
                if step % grad_accum_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.scheduler.step()
                sl_losses.append(sl_loss.item())
                rl_losses.append(rl_loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                summary['best_epoch'] = best_epoch
                summary['pi_loss'] = np.mean(pi_losses)
                summary['v_loss'] = np.mean(v_losses)
                reward_summary = self._evaluate_rew(self.val_loader)


                wandb.log(summary, e)
                if summary['ACC'] > best_ACC:
                    best_epoch = e
                    best_ACC = summary['ACC']
                    self._save_model('sl.pt', e)
                summary = self.evaluate(self.test_loader, prefix='TEST_')
                wandb.log(summary, e)
                pi_losses = []
                v_losses = []

        summary = self.evaluate(self.test_loader, prefix='TEST_')
        wandb.log(summary, e)
        self._save_model('last.pt', e)



    def _evaluate_rew(self, loader, actor, critic):
        summary = {}
        for k in self.args.k_list:
            summary['RW@'+str(k)] = 0.0
        summary['PERPLEXITY'] = 0.0
        self.model.eval()
        for step, (match_batch, user_history_batch) in tqdm.tqdm(enumerate(self.train_loader)):
            _, match_y_batch = match_batch
            match_y_batch = [feature.to(self.device).squeeze(0) for feature in match_y_batch]
            user_history_x_batch, user_history_y_batch = user_history_batch
            user_history_x_batch = [feature.to(self.device).squeeze(0) for feature in user_history_x_batch]
            user_history_y_batch = [feature.to(self.device).squeeze(0) for feature in user_history_y_batch]

            (ban_ids, item_ids, lane_ids, stat_ids, win_ids) = user_history_x_batch
            (win_labels, win_mask_labels, item_labels) = user_history_y_batch
            (_, pi_true, _) = match_y_batch
            N, _ = pi_true.shape
            turn = torch.argmax(pi_true, 1) - 1

            with torch.no_grad():
                pi_pred, _ = actor(user_history_x_batch)
                _, _, C = pi_pred.shape
                pi_pred = pi_pred[torch.arange(N, device=self.device), turn + 1, :]

                for k in self.args.k_list:
                    top_k_logits, top_k_actions = torch.topk(pi_pred, k)
                    top_k_probs = torch.exp(top_k_logits)
                    top_k_probs = top_k_probs / torch.sum(top_k_probs, 1, keepdim=True)
                    rew = torch.zeros_like(top_k_probs, device=self.device)

                    for k_idx in range(k):
                        k_actions = top_k_actions[:, k_idx]
                        next_item_ids = item_ids.clone()
                        next_item_ids[torch.arange(N, device=self.device), turn, -1] = k_actions
                        next_user_history_x_batch = (ban_ids, next_item_ids, lane_ids, stat_ids, win_ids)
                        _, q_pred = critic(next_user_history_x_batch)
                        q_pred = F.sigmoid(q_pred[:, 0, :])

                        import pdb
                        pdb.set_trace()

                        for i in range(N):
                            if turn[i] in [0, 2, 3, 6, 7]:
                                adv[k, i] = q_pred[i] - v_pred[i]
                            else:
                                adv[k, i] = (1 - q_pred[i]) - (1 - v_pred[i])

                        import pdb
                        pdb.set_trace()
            #    summary['HR@' + str(k)] += recall_at_k(pi_true, pi_pred, k) * N
            #    summary['NDCG@' + str(k)] += ndcg_at_k(pi_true, pi_pred, k) * N
            #summary['MRR'] += average_precision_at_k(pi_true, pi_pred, k=C) * N

