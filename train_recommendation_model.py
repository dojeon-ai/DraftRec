import tqdm
import wandb
import torch
import copy
import torch.nn.functional as F
from common.metrics import *
from common.train_utils import init_transformer_weights
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
        self.policy_criterion, self.value_criterion = self._initialize_criterion()
        self.optimizer = self._initialize_optimizer()

    def _initialize_model(self):
        model = Transformer(self.args, self.categorical_ids, self.device)
        model.apply(init_transformer_weights)
        model.value_head = init_transformer_weights(model.value_head, init_range=1.0)
        return model.to(self.device)

    def _initialize_criterion(self):
        policy_criterion = torch.nn.NLLLoss().to(self.device)
        value_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        return policy_criterion, value_criterion

    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self. args.weight_decay)
        return optimizer

    def _save_model(self, model_name, epoch):
        model_path = wandb.run.dir + '/' + str(model_name)
        torch.save({"state_dict": self.model.state_dict(),
                    "epoch": epoch}, model_path)

    def train(self):
        args = self.args
        policy_losses = []
        value_losses = []
        # evaluate the initial-run
        summary = self.evaluate(self.val_loader)
        wandb.log(summary, 0)
        # start training
        for e in range(1, args.epochs+1):
            print('[Epoch:%d]' % e)
            self.model.train()
            for x_batch, y_batch in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad()
                x_batch = [feature.to(self.device) for feature in x_batch]
                y_batch = [feature.to(self.device) for feature in y_batch]
                _, item_batch, win_batch = y_batch
                pi, v = self.model(x_batch)
                policy_loss = self.policy_criterion(pi, item_batch.long())
                value_loss = self.value_criterion(v, win_batch.float().unsqueeze(1))
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            if e % args.evaluate_every == 0:
                summary = self.evaluate(self.val_loader)
                summary['policy_loss'] = np.mean(policy_losses)
                summary['value_loss'] = np.mean(value_losses)
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
        pi_metrics = []
        for key in summary.keys():
            pi_metrics.append(key)
        summary['ACC'] = 0.0
        summary['MAE'] = 0.0
        summary['MSE'] = 0.0
        summary['ACC-LAST'] = 0.0
        summary['MAE-LAST'] = 0.0
        summary['MSE-LAST'] = 0.0

        # measure metrics
        UNK = 0
        num_evaluation = 0
        for x_batch, y_batch in tqdm.tqdm(loader):
            x_batch = [feature.to(self.device) for feature in x_batch]
            y_batch = [feature.to(self.device) for feature in y_batch]
            _, _, _, _, _, _, order_batch = x_batch
            user_batch, item_batch, win_batch = y_batch

            pi_true = torch.eye(self.num_items)[item_batch.long()].detach().cpu().numpy()
            v_true = win_batch.cpu().numpy()
            pi, v = self.model(x_batch)
            pi_pred = torch.exp(pi).detach().cpu().numpy()
            v_pred = F.sigmoid(v).squeeze(1).detach().cpu().numpy()

            # For recommendation, only test for the valid user in the training-set
            user_batch = user_batch.cpu().numpy()
            valid_user = (user_batch != UNK)
            pi_pred = pi_pred[valid_user]
            pi_true = pi_true[valid_user]
            for k in self.args.k_list:
                summary['HR@' + str(k)] += recall_at_k(pi_true, pi_pred, k) * np.sum(valid_user)
                summary['NDCG@' + str(k)] += ndcg_at_k(pi_true, pi_pred, k) * np.sum(valid_user)
            summary['MRR'] += average_precision_at_k(pi_true, pi_pred, pi_true.shape[1]) * np.sum(valid_user)
            num_evaluation += np.sum(valid_user)

            # For win-rate prediction, measure metrics of [s_1, .., s_T] and [s_T] separately
            order_batch = order_batch.cpu().numpy()
            summary['ACC'] += np.sum((v_pred >= 0.5) == v_true)
            summary['MAE'] += np.sum(np.abs(v_pred - v_true))
            summary['MSE'] += np.sum(np.square(v_pred - v_true))

            last_board = (order_batch == 9)
            v_pred = v_pred[last_board]
            v_true = v_true[last_board]
            summary['ACC-LAST'] += np.sum((v_pred >= 0.5) == v_true)
            summary['MAE-LAST'] += np.sum(np.abs(v_pred - v_true))
            summary['MSE-LAST'] += np.sum(np.square(v_pred - v_true))

        for metric, value in summary.items():
            if metric in pi_metrics:
                summary[metric] = (value / num_evaluation) * 100
            elif metric in ['ACC', 'MAE', 'MSE']:
                summary[metric] = (value / len(loader.dataset))
            elif metric in ['ACC-LAST', 'MAE-LAST', 'MSE-LAST']:
                summary[metric] = (value / len(loader.dataset)) * 10

        return summary
