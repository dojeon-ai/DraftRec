import argparse
import os
import sys
import warnings
import pickle
from dotmap import DotMap
from typing import List

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
    if args.model_type in ['random', 'pop', 'nmf']:
        args.train_dataloader_type = 'interaction'
        args.trainer_type = 'interaction'
        
    elif args.model_type in ['sasrec', 'sasrec_moba']:
        args.train_dataloader_type = 'sequential'
        args.trainer_type = 'sequential'
        
    elif args.model_type in ['optmatch', 'draftrec']:
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
    with open(dataset_path + '/user_history_dict.pickle', 'rb') as f:
        user_history_dict = pickle.load(f)
    print('[Finish loading the dataset]')
    
    # TODO: remove this with categorical ids
    args.num_champions = 200
    args.num_roles = 10
    args.num_teams = 10
    args.num_outcomes = 10
    args.num_stats = 43    
    
    # DataLoader
    train_dataloader, val_dataloader, test_dataloader = init_dataloader(args, match_df, user_history_dict)

    # Model
    model = init_model(args)
    
    # Trainer
    trainer = init_trainer(args, train_dataloader, val_dataloader, test_dataloader, model)
    trainer.train()
    
    #for batch in train_dataloader:
    #    model(batch, train_dataloader, val_dataloader, test_dataloader, model)
    
if __name__ == "__main__":
    main()



