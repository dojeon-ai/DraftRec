import torch
import torch.nn as nn
import torch.nn.functional as F


#class Concat(nn.Module):
#    def __init__(self, args, categorical_ids):
#        super(Concat, self).__init__()
#        self.num_users = len(categorical_ids['user'])
#        self.num_items = len(categorical_ids['champion'])
#
#        self.embedding_dim = args.embedding_dim
#        self.num_hidden_layers = args.num_hidden_layers
#        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
#        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
#
#        phi = []
#        hidden_dim = self.embedding_dim * 2
#        for _ in range(self.num_hidden_layers):
#            phi.append(nn.Linear(hidden_dim, self.embedding_dim))
#            phi.append(nn.ReLU())
#            hidden_dim = self.embedding_dim
#        self.phi = nn.ModuleList(phi)
#        self.h = nn.Linear(hidden_dim, 1)
#
#    def forward(self, x):
#        user_ids, item_ids = x
#        p = self.user_embedding(user_ids)
#        q = self.item_embedding(item_ids)
#        z = torch.cat((p, q), 1)
#        for layer in self.phi:
#            z = layer(z)
#        logit = self.h(z)
#        return logit


class NMF(nn.Module):
    def __init__(self, args, categorical_ids):
        super(NMF, self).__init__()
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])

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
    def __init__(self, args, categorical_ids):
        super(DMF, self).__init__()
        pass

    def forward(self, x):
        pass


class POP(nn.Module):
    def __init__(self, args, categorical_ids):
        super(POP, self).__init__()
        pass

    def forward(self, x):
        pass