from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class NMF(BaseModel):
    def __init__(self, args):
        super(NMF, self).__init__(args)
        self.num_users = args.num_users 
        self.num_items = args.num_champions

        self.embedding_dim = args.hidden_units
        self.num_hidden_layers = args.num_hidden_layers
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        phi = []
        for _ in range(self.num_hidden_layers):
            phi.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            phi.append(nn.ReLU())
        self.phi = nn.ModuleList(phi)
        self.h = nn.Linear(self.embedding_dim, 1)
        

    @classmethod
    def code(cls):
        return 'nmf'

    def forward(self, batch):
        user_ids = batch['user_idx']
        champion_ids = batch['champion_idx']
        p = self.user_embedding(user_ids)
        q = self.item_embedding(champion_ids)
        z = (p * q)
        for layer in self.phi:
            z = layer(z)
        logit = self.h(z)
        logit = F.sigmoid(logit)
        return logit


class DMF(BaseModel):
    def __init__(self, args):
        super(DMF, self).__init__(args)
        self.num_users = args.num_users 
        self.num_items = args.num_champions
        self.embedding_dim = args.hidden_units
        self.num_hidden_layers = args.num_hidden_layers
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        phi_user = []
        phi_item = []
        for _ in range(self.num_hidden_layers):
            phi_user.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            phi_user.append(nn.ReLU())
        for _ in range(self.num_hidden_layers):
            phi_item.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            phi_item.append(nn.ReLU())
        self.phi_user = nn.ModuleList(phi_user)
        self.phi_item = nn.ModuleList(phi_item)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.min_prob = args.min_prob

    @classmethod
    def code(cls):
        return 'dmf'

    def forward(self, batch):
        user_ids = batch['user_idx']
        champion_ids = batch['champion_idx']
        p = self.user_embedding(user_ids)
        q = self.item_embedding(champion_ids)
        for layer in self.phi_user:
            p = layer(p)
        for layer in self.phi_item:
            q = layer(q)

        logit = self.cos(p, q)
        logit = torch.where(logit < 0, torch.Tensor([float(self.min_prob)]).to(self.args.device), logit)
        logit = logit.reshape(-1, 1)
        return logit




class POP(BaseModel):
    def __init__(self, args):
        super(POP, self).__init__(args)
        self.num_champions = args.num_champions
        self.null = nn.Linear(1, 1)

    @classmethod
    def code(cls):
        return 'pop'

    def forward(self, batch, pop_dict):

        args = self.args
        num_champions = args.num_champions
        num_turns = args.num_turns
        batch_size = batch['user_ids'].shape[0]

        turn_idx = copy.deepcopy(batch['turn'])
        turn_idx[turn_idx > num_turns] = 1
        turn_idx = turn_idx - 1

        user_ids = torch.gather(batch['user_ids'], 1, turn_idx).repeat_interleave(num_champions)  # (N*C)
        champion_ids = torch.arange(num_champions).repeat(batch_size)  # (N*C)

        N = len(user_ids)
        logit = torch.zeros((N, 1))
        for idx, (user_id, champion_id) in enumerate(zip(user_ids, champion_ids)):
            rank = pop_dict[user_id.item()].tolist().index(champion_id.item())
            logit[idx] = 1 - rank/num_champions

        logit = logit.reshape(batch_size, num_champions)

        logit[torch.arange(batch_size).unsqueeze(1), batch['bans'].cpu().numpy()] = 1e-10

        return logit
"""
class POP(BaseModel):
    def __init__(self, args):
        super(POP, self).__init__(args)
        self.num_champions = args.num_champions
        self.null = nn.Linear(1, 1)

    @classmethod
    def code(cls):
        return 'pop'

    def forward(self, batch):
        user_ids = batch['user_idx']
        champion_ids = batch['champion_idx']
        pop_dict= batch['pop_dict']
        N = len(user_ids)
        logit = torch.zeros((N, 1))
        for idx, (user_id, champion_id) in enumerate(zip(user_ids, champion_ids)):
            rank = pop_dict[user_id.item()].tolist().index(champion_id.item())
            logit[idx] = 1 - rank/self.num_champions

        return logit
"""
class SPOP(BaseModel):
    def __init__(self, args):
        super(SPOP, self).__init__(args)
        num_turns = args.num_turns
        num_champions = args.num_champions
        num_roles = args.num_roles
        num_teams = args.num_teams
        num_outcomes = args.num_outcomes
        num_stats = args.num_stats
        max_seq_len = args.max_seq_len
        self.null = nn.Linear(1, 1)
        
    @classmethod
    def code(cls):
        return 'spop'

    def forward(self, batch):
        

        args = self.args

        num_champions = args.num_champions
        num_turns = args.num_turns
        batch_size = batch['user_champions'].shape[0]
        sequence_len = batch['user_champions'].shape[2]

        user_champions = batch['user_champions'] #(512, 10, 100)

        turn_idx = copy.deepcopy(batch['turn'])
        turn_idx[turn_idx > num_turns] = 1
        turn_idx = turn_idx - 1
        turn_idx = turn_idx.repeat(1, sequence_len).unsqueeze(1)

        user_history = torch.gather(user_champions, 1, turn_idx).squeeze(1)

        pi_logit = torch.zeros((batch_size, num_champions))
        for idx in range(batch_size):
            current_user_history = user_history[idx]
            pi_logit[idx] = torch.bincount(current_user_history[current_user_history != args.PAD], minlength=num_champions)
        
        # banned champion should not be picked
        pi_logit[torch.arange(batch_size)[:, None], batch['bans']] = -1
        
        #pi = F.log_softmax(pi_logit, dim=-1)  # log-probability is passed to NLL-Loss
        return pi_logit