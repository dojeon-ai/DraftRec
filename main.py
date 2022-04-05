import argparse
import os
import sys
import warnings
import json
import tqdm
import numpy as np
import pandas as pd
from dotmap import DotMap
from typing import List
from multiprocessing import Manager

from arguments import Parser
from src.common.data_utils import *
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
        
    # Dataset
    print('[Start loading the dataset]')
    dataset_path = args.local_data_folder + '/' + args.dataset_type

    match_df = {}
    match_df['train'] = pd.read_csv(dataset_path + '/train.csv', index_col=0)
    match_df['val'] =  pd.read_csv(dataset_path + '/val.csv', index_col=0)
    match_df['test'] =  pd.read_csv(dataset_path + '/test.csv', index_col=0)

    user_history_array = np.load(dataset_path + '/user_history.npy', mmap_mode='r+')
    # Load data in memory if your memory (>30Gb)
    # user_history_array = read_large_array(dataset_path + '/user_history.npy')

    with open(dataset_path + '/categorical_ids.json', 'r') as f:
        categorical_ids = json.load(f)
    with open(dataset_path + '/feature_to_array_idx.json', 'r') as f:
        feature_to_array_idx = json.load(f)

    print('[Finish loading the dataset]')
    args.num_champions = len(categorical_ids['champion'])
    args.num_roles = len(categorical_ids['role'])
    args.num_teams = len(categorical_ids['team'])
    args.num_outcomes = len(categorical_ids['win'])
    args.num_stats = len(categorical_ids['stats'])

    # DataLoader
    train_dataloader, val_dataloader, test_dataloader = init_dataloader(args, 
                                                                        match_df, 
                                                                        user_history_array, 
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



