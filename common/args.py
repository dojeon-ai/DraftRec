import argparse
import ast


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError
    return v


def add_interaction_arguments(parser):
    # model
    parser.add_argument('--model_type', choices=['concat', 'dot'], default='dot')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    # train
    parser.add_argument('--num_negatives', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    # eval
    parser.add_argument('--k_list', type=str2list, default=[1, 5, 10])
    parser.add_argument('--evaluate_every', type=int, default=5)

    return parser


def add_recommendation_arguments(parser):
    # model
    parser.add_argument('--embedding_dim', type=int, default=64)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=2)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    parser.add_argument('--weight_init_range', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    # eval
    parser.add_argument('--k_list', type=str2list, default=[1, 5, 10])
    parser.add_argument('--evaluate_every', type=int, default=1)
    return parser
