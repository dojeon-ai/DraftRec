import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, n_position, d_hid):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: type:(torch.Tensor) shape:(S:seq_len, N:batch_size, I:input_dim)
        :return x: type:(torch.Tensor) shape:(S, N, I:input_dim)
        """
        S, N, I = x.shape
        x = self.w2(F.relu(self.w1(x.view(S*N, I)))).view(S, N, I)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.feed_forward = FeedForward(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        dropout=dropout)
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        """
        :param x: type:(torch.Tensor) shape:(S:seq_len, N:batch_size, I:input_dim)
        :return x: type:(torch.Tensor) shape:(S, N, I)
        """
        y, attn_weights = self.attention(x, x, x)
        x = x+y
        x = self.norm1(x)

        y = self.feed_forward(x)
        x = x+y
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, args, categorical_ids):
        super(Transformer, self).__init__()
        self.args = args
        self.num_teams = 3  # [CLS, BLUE, RED]
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.num_lanes = len(categorical_ids['lane'])
        self.num_version = len(categorical_ids['version'])
        self.num_position = 10

        self.embedding_dim = args.embedding_dim
        self.team_embedding = nn.Embedding(self.num_teams, self.embedding_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.lane_embedding = nn.Embedding(self.num_lanes, self.embedding_dim)
        self.version_embedding = nn.Embedding(self.num_version, self.embedding_dim)
        self.position_embedding = PositionalEncoding(self.num_position, self.embedding_dim)

        encoder = []
        for _ in range(self.args.num_hidden_layers-1):
            encoder.append(EncoderLayer(self.embedding_dim,
                                        self.embedding_dim,
                                        self.args.num_heads,
                                        self.args.dropout))
        self.encoder = nn.ModuleList(encoder)

    def forward(self, x):
        team, ban, user, item, lane, version, order = x

        team = self.team_embedding(team.long())
        user = self.user_embedding(user.long())
        item = self.item_embedding(item.long())
        lane = self.lane_embedding(lane.long())

        x = team + user + item + lane
        x = self.position_embedding(x)
        # TODO: append CLS token

        for layer in self.encoder:
            x = layer(x)


        import pdb
        pdb.set_trace()
        #position = self.position_embedding

        #input =



        import pdb
        pdb.set_trace()
        pass


