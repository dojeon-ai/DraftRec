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
    default = {'--exp_name': 'test_nn'}

    seeds = ['0']
    temps = ['nn_lol', 'nn_dota']
    lrs = ['0.01', '0.001', '0.0001']
    
    num_devices = 4
    num_exp_per_device = 2
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for seed, temp, lr in itertools.product(*[seeds, temps, lrs]):
        exp = copy.deepcopy(default)       
        exp['--seed'] = seed
        exp['--template'] = temp
        exp['--lr'] = lr
        
        device_idx = device % num_devices
        exp['--device'] = 'cuda:' + str(int(device_idx))
        
        experiments.append(exp)
        device += 1
        
    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)    
    pool.close()
    
