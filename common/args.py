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
    parser.add_argument('--model_type', choices=['pop', 'nmf', 'dmf'], default='nmf')
    parser.add_argument('--target_type', choices=['implicit', 'explicit'], default='explicit')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=0)
    # train
    parser.add_argument('--num_negatives', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--min_prob', type=float, default=1e-6)
    return parser


def add_user_rec_arguments(parser):
    # data
    parser.add_argument('--max_seq_len', type=int, default=20)
    parser.add_argument('--mask_prob', type=float, default=0.3, help='only used in bert model')
    parser.add_argument('--use_game_specific_info', type=str2bool, default=True)
    # model
    parser.add_argument('--model_type', choices=['spop', 'sas', 'bert'], default='sas')
    parser.add_argument('--embedding_dim', type=int, default=512)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=8)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lmbda', type=float, default=0.0)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    return parser


def add_draft_rec_arguments(parser):
    # data
    parser.add_argument('--max_seq_len', type=int, default=20)
    parser.add_argument('--use_game_specific_info', type=str2bool, default=True)
    # model
    parser.add_argument('--embedding_dim', type=int, default=512)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=8)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    return parser


def add_reward_model_arguments(parser):
    # data
    parser.add_argument('--mask_item_prob', type=float, default=0.1)
    parser.add_argument('--label_smooth', type=float, default=0.0)
    # model
    parser.add_argument('--embedding_dim', type=int, default=512)  # [64, 256]
    parser.add_argument('--num_hidden_layers', type=int, default=2)  # [2]
    parser.add_argument('--num_heads', type=int, default=8)  # [2]
    parser.add_argument('--dropout', type=float, default=0.1)  # [0.0, 0.1, 0.2]
    # train
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    return parser
