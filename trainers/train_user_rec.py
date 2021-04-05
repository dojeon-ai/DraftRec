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
        summary = self.evaluate(self.val_loader)
        best_HR1 = 0
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
                (win_labels, win_mask_labels, item_labels) = y_batch

                pi, v = self.model(x_batch)
                N, S, C = pi.shape
                policy_loss = self.policy_criterion(pi.reshape(N*S, C), item_labels.reshape(-1))
                value_loss = self.value_criterion(v[win_mask_labels == 1].squeeze(-1), win_labels[win_mask_labels == 1])
                loss = (1-args.lmbda) * policy_loss + args.lmbda * value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad)
                self.optimizer.step()
                # self.scheduler.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                if summary['HR@1'] > best_HR1:
                    best_epoch = e
                    best_HR1 = summary['HR@1']
                summary['best_epoch'] = best_epoch
                summary['policy_loss'] = np.mean(policy_losses)
                summary['value_loss'] = np.mean(value_losses)
                wandb.log(summary, e)
                # self._save_model('model.pt', e)

    def evaluate(self, loader):
        self.model.eval()
        # initialize metrics
        summary = {}
        for k in self.args.k_list:
            summary['HR@'+str(k)] = 0.0
            summary['NDCG@'+str(k)] = 0.0
        summary['MRR'] = 0.0
        for metric in ['ACC', 'MAE', 'MSE']:
            summary[metric] = 0.0

        total_num_valid_users = 0
        for x_batch, y_batch in tqdm.tqdm(loader):
            x_batch = [feature.to(self.device) for feature in x_batch]
            y_batch = [feature.to(self.device) for feature in y_batch]
            (team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids) = x_batch
            (win_labels, item_labels) = y_batch  # [N,S], [N,S]

            item_labels, item_labels_idx = torch.max(item_labels, 1)  # [N]
            item_labels = torch.eye(self.num_items)[item_labels].detach().cpu().numpy()  # [N, C]
            UNK = 3
            user_x_batch, user_y_batch = [], []
            for batch_idx, seq_idx in enumerate(item_labels_idx):
                user_idx = user_ids[batch_idx][seq_idx].item()
                history_idx = history_ids[batch_idx][seq_idx].item()
                if user_idx != UNK:
                    user_x, user_y = self.train_loader.dataset.get_item_with_history_idx(user_idx, history_idx)
                    user_x_batch.append(user_x)
                    user_y_batch.append(user_y)

            user_x_batch = [torch.stack(feature).to(self.device) for feature in list(map(list, zip(*user_x_batch)))]
            user_y_batch = [torch.stack(feature).to(self.device) for feature in list(map(list, zip(*user_y_batch)))]
            (_, _, user_item_labels) = user_y_batch
            pi, v = self.model(user_x_batch)
            pi = torch.exp(pi)
            v = F.sigmoid(v)
            # select the right sequence (pi: [last])
            N, S, C = pi.shape
            pi = pi[torch.arange(N, device=self.device), -1, :].detach().cpu().numpy()
            user_item_labels = user_item_labels[torch.arange(N, device=self.device), -1].detach().cpu().numpy()
            user_item_labels = np.eye(self.num_items)[user_item_labels]

            # Perform evaluation
            for k in self.args.k_list:
                summary['HR@' + str(k)] += recall_at_k(user_item_labels, pi, k) * N
                summary['NDCG@' + str(k)] += ndcg_at_k(user_item_labels, pi, k) * N
            summary['MRR'] += average_precision_at_k(user_item_labels, pi, pi.shape[1]) * N
            total_num_valid_users += N

        for metric, value in summary.items():
            summary[metric] = (value / total_num_valid_users)
        return summary




















