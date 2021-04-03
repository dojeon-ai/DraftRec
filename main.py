import argparse
import os
import warnings
import random
import pickle
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from common.args import *
# Dataset
from dataset.interaction_dataset import InteractionDataset
from dataset.user_rec_dataset import UserRecDataset
from dataset.context_rec_dataset import ContextRecDataset
from dataset.match_eval_dataset import MatchEvalDataset


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='arguments for reward model')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--op', default='train_context_rec', choices=['train_interaction', 'train_user_rec', 'train_context_rec'])
    parser.add_argument('--data_dir', type=str, default='./data')
    #parser.add_argument('--data_dir', type=str, default='/home/nas1_userC/hojoonlee/draftRec/data')
    parser.add_argument('--interaction_path', type=str, default='/interaction_data.pickle')
    parser.add_argument('--match_path', type=str, default='/match_data.pickle')
    parser.add_argument('--user_history_path', type=str, default='/user_history_data.pickle')
    parser.add_argument('--dict_path', type=str, default='/categorical_ids.pickle')
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for dataloader")
    parser.add_argument('--gpu', type=int, default=-1, help='index of the gpu device to use (cpu if -1)')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999))
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--user', type=str, default='hj', choices=['hj', 'dy', 'hs', 'bk'])
    parser.add_argument('--num_threads', type=int, default=1)
    args = parser.parse_known_args()[0]

    #############
    ## Configs ##
    #############
    print('[INITIALIZE ENVIRONMENTAL CONFIGS]')
    if args.user == 'hj':
        wandb_id = '96022b49a4e5c639895ba1e229022e087f79c84a'
        user_id = 'joonleesky'
    else:
        raise NotImplementedError
    wandb.login()
    wandb.init(project='draftRec')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(args.gpu))
    device = torch.device('cuda' if args.gpu != -1 else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    ################
    ## DataLoader ##
    ################
    # Initialize raw-data & loader
    print('[LOAD DATA]')
    interaction_data_path = args.data_dir + args.interaction_path
    with open(interaction_data_path, 'rb') as f:
        interaction_data = pickle.load(f)

    match_data_path = args.data_dir + args.match_path
    with open(match_data_path, 'rb') as f:
        match_data = pickle.load(f)

    user_history_data_path = args.data_dir + args.user_history_path
    with open(user_history_data_path, 'rb') as f:
        user_history_data = pickle.load(f)

    dict_path = args.data_dir + args.dict_path
    with open(dict_path, 'rb') as f:
        categorical_ids = pickle.load(f)

    #############
    ## Trainer ##
    #############
    # Initialize data & trainer func
    print('[INITIALIZE DATA LOADER & TRAINER FUNC]')
    if args.op == 'train_interaction':
        parser = add_interaction_arguments(parser)
        args = parser.parse_args()
        wandb.config.update(args)
        train_data = InteractionDataset(args,
                                        interaction_data['train'],
                                        categorical_ids,
                                        is_train=True)
        from trainers.train_interaction import InteractionTrainer as Trainer

    elif args.op == 'train_user_rec':
        parser = add_user_rec_arguments(parser)
        args = parser.parse_args()
        wandb.config.update(args)
        train_data = UserRecDataset(args,
                                    user_history_data,
                                    categorical_ids)
        from trainers.train_user_rec import UserRecTrainer as Trainer

    elif args.op == 'train_context_rec':
        parser = add_context_rec_arguments(parser)
        args = parser.parse_args()
        wandb.config.update(args)
        train_data = ContextRecDataset(args,
                                       match_data['train'],
                                       categorical_ids)
        from trainers.train_context_rec import ContextRecTrainer as Trainer
    else:
        raise NotImplementedError

    val_data = MatchEvalDataset(args,
                                match_data['val'][:1000],
                                categorical_ids)
    test_data = MatchEvalDataset(args,
                                 match_data['test'][:1000],
                                 categorical_ids)

    del interaction_data
    del match_data

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # Start training
    print('[START TRAINING]')
    trainer = Trainer(args, train_loader, val_loader, test_loader, categorical_ids, device)
    trainer.train()
