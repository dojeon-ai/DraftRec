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
    parser.add_argument('--model_type', choices=['nmf', 'dmf'], default='nmf')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    # train
    parser.add_argument('--num_negatives', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    # eval
    parser.add_argument('--k_list', type=str2list, default=[1, 5, 10])
    parser.add_argument('--evaluate_every', type=int, default=5)

    return parser


def add_user_rec_arguments(parser):
    # data
    parser.add_argument('--max_seq_len', type=int, default=15)
    parser.add_argument('--mask_prob', type=float, default=0.3)
    # model
    parser.add_argument('--model_type', choices=['sas', 'bert'], default='bert')
    parser.add_argument('--embedding_dim', type=int, default=512)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=8)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--v_start', type=float, default=50, help='epoch to start training value')
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    # eval
    parser.add_argument('--k_list', type=str2list, default=[1, 5, 10])
    parser.add_argument('--evaluate_every', type=int, default=5)
    return parser


def add_context_rec_arguments(parser):
    # data
    parser.add_argument('--mask_prob', type=float, default=0.8)
    # model
    parser.add_argument('--embedding_dim', type=int, default=512)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=8)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--v_start', type=float, default=50, help='epoch to start training value')
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    # eval
    parser.add_argument('--k_list', type=str2list, default=[1, 5, 10])
    parser.add_argument('--evaluate_every', type=int, default=5)
    return parser


def add_draft_rec_arguments(parser):
    # data
    parser.add_argument('--max_seq_len', type=int, default=15)
    # model
    parser.add_argument('--embedding_dim', type=int, default=512)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=8)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--v_start', type=float, default=0, help='epoch to start training value')
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    # eval
    parser.add_argument('--k_list', type=str2list, default=[1, 5, 10])
    parser.add_argument('--evaluate_every', type=int, default=1)
    return parser

