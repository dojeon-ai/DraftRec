import subprocess
import argparse
import json
import copy
import itertools
from multiprocessing import Pool


def run_experiment(experiment):
    cmd = ['python', 'main.py']
    for key, value in experiment.items():
        cmd.append(key)
        cmd.append(value)
    return subprocess.check_output(cmd)


if __name__ == '__main__':
    default = {'--exp_name': '1029_tune_nac_lol'}

    seeds = ['0']
    temps = ['nac_lol']
    lrs = ['0.01', '0.001']
    wds = ['0.01', '0.001', '0.0001']
    hus = ['32', '64', '128']
    
    num_devices = 3
    num_exp_per_device = 2
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for seed, temp, lr, wd, hu in itertools.product(*[seeds, temps, lrs, wds, hus]):
        exp = copy.deepcopy(default)       
        exp['--seed'] = seed
        exp['--template'] = temp
        exp['--lr'] = lr
        exp['--weight_decay'] = wd
        exp['--hidden_units'] = hu
        
        exp['--device'] = 'cuda:' + str(int(device % num_devices))
        experiments.append(exp)
        device += 1
        
        
        
    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)    
    pool.close()
    
