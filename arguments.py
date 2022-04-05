import argparse
import yaml
import os
from copy import deepcopy


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

class Parser:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def set_template(self, configs):
        template_name = configs['template']
        assert template_name is not None, 'template is not given'
        template = yaml.safe_load(open(os.path.join('templates', f'{template_name}.yaml')))
        # overwrite_with_non_nones
        for k, v in configs.items():
            if v is None:
                configs[k] = template[k]
        return configs

    def parse(self):
        """
        1. initialize null arguments (parser should not contain default arguments)
        2. fill the arguments with cmd interface
        3. fill the arguments from templates
        priority: cmd > template
        """
        configs = {}
        configs.update(self.parse_top())
        configs.update(self.parse_dataloader())
        configs.update(self.parse_trainer())
        configs.update(self.parse_model())

        configs = self.set_template(configs)
        return configs

    def parse_top(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--template', type=str)
        parser.add_argument('--seed', type=float, help='Random seed to initialize the random state')
        parser.add_argument('--pilot', type=str2bool, help='If true, run the program in minimal amount to check for errors')
        parser.add_argument('--local_data_folder', type=str, help='Folder that contains raw/preprocessed data')
        parser.add_argument('--wandb_project_name', type=str, help='Project name for wandb')
        parser.add_argument('--exp_name', type=str, help='experiment name to identify')
        parser.add_argument('--dataset_type', type=str, choices=['lol','dota'], help='select the dataset to use for the experiment')
        parser.add_argument('--model_type', type=str, choices=[ 'lr', 'nn', 'draftrec'], help='Selects the model for the experiment')
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_dataloader(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--train_dataloader_type', type=str, choices=['match'], help='Selects the trainer for the experiment')
        parser.add_argument('--val_dataloader_type', type=str, choices=['match'], help='Selects the trainer for the experiment')
        parser.add_argument('--test_dataloader_type', type=str, choices=['match'], help='Selects the trainer for the experiment')
        parser.add_argument('--train_batch_size', type=int, help='Batch size for training')
        parser.add_argument('--val_batch_size', type=int, help='Batch size for validation')
        parser.add_argument('--test_batch_size', type=int, help='Batch size for test')
        parser.add_argument('--max_seq_len', type=int, help="maximum sequence length")
        parser.add_argument('--num_special_tokens', type=int, help="number of special tokens [PAD, MASK, CLS, UNK]")
        parser.add_argument('--num_turns', type=int, help="number of turns for each match")
        parser.add_argument('--num_champions', type=int, help="number of stats for each user in the match")
        parser.add_argument('--num_roles', type=int, help="number of stats for each user in the match")
        parser.add_argument('--num_teams', type=int, help="number of stats for each user in the match")
        parser.add_argument('--num_outcomes', type=int, help="number of stats for each user in the match")
        parser.add_argument('--num_stats', type=int, help="number of stats for each user in the match")
        parser.add_argument('--num_negatives', type=int, help="number of stats for each user in the match")
        parser.add_argument('--window_size', type=int, help="window size to slide over the user's entire item sequences to obtain subsequences for training")
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_trainer(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--trainer_type', type=str, choices=['match'], help='Selects the trainer for the experiment')
        parser.add_argument('--device', type=str)
        parser.add_argument('--use_parallel', type=str2bool, help='If true, the program uses all visible cuda devices with DataParallel')
        parser.add_argument('--num_workers', type=int)
        # optimizer #
        parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, help='l2 regularization')
        parser.add_argument('--momentum', type=float, help='sgd momentum')
        parser.add_argument('--lmbda', type=float, help='lambda to balance the policy loss and value loss')
        # regularization
        parser.add_argument('--clip_grad_norm', type=float)
        parser.add_argument('--dropout', type=float, help='Dropout probability')
        # epochs #
        parser.add_argument('--num_epochs', type=int, help='Maximum number of epochs to run')
        # evaluation #
        parser.add_argument('--metric_ks', nargs='+', type=int, help='list of k for NDCG@k and HR@k')
        parser.add_argument('--best_metric', type=str, help='This metric will be used to compare and determine the best model')
        
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_model(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--model_init_seed', type=int, help='Seed used to initialize the model parameters')
        parser.add_argument('--hidden_units', type=int, help='Hidden dimension size')
        parser.add_argument('--num_blocks', type=int, help='Number of transformer layers')
        parser.add_argument('--head_type', type=str, choices=['linear', 'dot'], help='Prediction heads on top of logits')
        parser.add_argument('--num_heads', type=int, help='Number of attention heads')
        parser.add_argument('--num_hidden_layers', type=int, help='Number of hidden layers for FC models')        
        parser.add_argument('--min_prob', type=int, help='minimum probability for DMF')  
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)
