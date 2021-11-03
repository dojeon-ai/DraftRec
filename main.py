import argparse
import os
import sys
import warnings
import pickle
import tqdm
import numpy as np
from dotmap import DotMap
from typing import List
from multiprocessing import Manager

from arguments import Parser
from src.dataloaders import init_dataloader
from src.models import init_model
from src.trainers import init_trainer

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def main(sys_argv: List[str] = None):
    # Parser
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    configs = Parser(sys_argv).parse()
    args = DotMap(configs, _dynamic=False)
    # Registry
    if args.model_type in ['random', 'spop', 'pop', 'nmf']:
        args.train_dataloader_type = 'interaction'
        args.trainer_type = 'interaction'

    elif args.model_type in ['sasrec', 'sasrec_moba', 'lr', 'hoi', 'nac', 'optmatch', 'draftrec']:
        if args.use_full_info:
            args.train_dataloader_type = 'full_match'
        else:
            args.train_dataloader_type = 'match'
        args.trainer_type = 'match'
        
    else:
        raise NotImplementedError
        
    args.val_dataloader_type = 'match'
    args.test_dataloader_type = 'match'
        
    # Dataset
    print('[Start loading the dataset]')
    dataset_path = args.local_data_folder + '/' + args.dataset_type
    with open(dataset_path + '/match_df.pickle', 'rb') as f:
        match_df = pickle.load(f)
    with open(dataset_path + '/categorical_ids.pickle', 'rb') as f:
        categorical_ids = pickle.load(f)
    with open(dataset_path + '/feature_to_array_idx.pickle', 'rb') as f:
        feature_to_array_idx = pickle.load(f)
    with open(dataset_path + '/user_id_to_array_idx.pickle', 'rb') as f:
        user_id_to_array_idx = pickle.load(f)
    user_history_array = np.load(dataset_path + '/user_history_array.npy')
    print('[Finish loading the dataset]')
    
    # TODO: remove this with categorical ids
    args.num_champions = len(categorical_ids['champion'])
    args.num_roles = len(categorical_ids['role'])
    args.num_teams = len(categorical_ids['team'])
    args.num_outcomes = len(categorical_ids['win'])
    if args.dataset_type == 'lol':
        args.num_stats = 43    
    elif args.dataset_type == 'dota':
        args.num_stats = 26
        
    # DataLoader
    train_dataloader, val_dataloader, test_dataloader = init_dataloader(args, 
                                                                        match_df, 
                                                                        user_history_array, 
                                                                        user_id_to_array_idx, 
                                                                        feature_to_array_idx)

    # Model
    model = init_model(args)
    
    # Trainer
    trainer = init_trainer(args, 
                           train_dataloader, 
                           val_dataloader, 
                           test_dataloader, 
                           model)
    trainer.train()
    
if __name__ == "__main__":
    main()



