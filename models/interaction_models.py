import torch
import torch.nn as nn
import torch.nn.functional as F


class NMF(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(NMF, self).__init__()
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.device = device

        self.embedding_dim = args.embedding_dim
        self.num_hidden_layers = args.num_hidden_layers
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        phi = []
        for _ in range(self.num_hidden_layers):
            phi.append(nn.Linear(self.embedding_dim, self.embedding_dim))
            phi.append(nn.ReLU())
        self.phi = nn.ModuleList(phi)
        self.h = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        user_ids, item_ids = x
        p = self.user_embedding(user_ids)
        q = self.item_embedding(item_ids)
        z = (p * q)
        for layer in self.phi:
            z = layer(z)
        logit = self.h(z)
        return logit


class DMF(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(DMF, self).__init__()
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.device = device

        self.embedding_dim = args.embedding_dim
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

    def forward(self, x):
        user_ids, item_ids = x
        p = self.user_embedding(user_ids)
        q = self.item_embedding(item_ids)
        for layer in self.phi_user:
            p = layer(p)
        for layer in self.phi_item:
            q = layer(q)

        logit = self.cos(p, q)
        logit = torch.where(logit < 0, torch.Tensor([self.min_prob]).to(self.device), logit)
        logit = logit.reshape(-1, 1)
        return logit


class POP(nn.Module):
    def __init__(self, pop_dict, categorical_ids, device):
        super(POP, self).__init__()
        self.pop_dict = pop_dict
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.device = device
        self.null = nn.Linear(1, 1)

    def forward(self, x):
        user_ids, item_ids = x
        N = len(user_ids)
        logit = torch.zeros((N, 1), device=self.device)
        for idx, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            rank = self.pop_dict[user_id.item()].tolist().index(item_id.item())
            logit[idx] = 1 - rank/self.num_items

        return logit