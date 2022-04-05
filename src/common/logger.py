import wandb
import torch
import os
import json

class LoggerService(object):
    def __init__(self, args):
        self.train_logger = WandbLogger(prefix='train')
        self.val_logger = WandbLogger(prefix='val')
        self.test_logger = WandbLogger(prefix='test') 
        
        project_name = args.wandb_project_name
        exp_name = args.exp_name
        
        assert project_name is not None and exp_name is not None
        wandb.init(project=project_name, config=args, settings=wandb.Settings(start_method="fork"))        
        self.model_path = wandb.run.dir + '/model.pth'
        self.config_path = wandb.run.dir + '/config.json'
        with open(self.config_path, 'w') as f:
            json.dump(args.toDict(), f)
        
    def complete(self, log_data):
        self.train_logger.complete(**log_data)
        self.val_logger.complete(**log_data)
        self.test_logger.complete(**log_data)

    def log_train(self, log_data):
        self.train_logger.log(**log_data)

    def log_val(self, log_data):
        self.val_logger.log(**log_data)

    def log_test(self, log_data):
        self.test_logger.log(**log_data)
        
    def save_state_dict(self, state_dict):
        torch.save(state_dict, self.model_path)
    
    def load_state_dict(self):
        return torch.load(self.model_path)
            
            
class WandbLogger(object):
    def __init__(self, prefix=''):
        self.prefix = prefix

    def log(self, *args, **kwargs):
        try:
            step = kwargs['step']
        except:
            raise ValueError
                                                    
        log_dict = {}
        for k, v in kwargs.items():
            if k == 'step':
                continue
            elif k == 'epoch':
                log_dict[k] = v
            else:
                log_dict[self.prefix + '_' + k] = v
                
        wandb.log(log_dict, step=step)

    def complete(self, *args, **kwargs):
        wandb.log({})  # ensure to not miss the last log



class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)