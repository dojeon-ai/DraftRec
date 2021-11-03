import torch.utils.data as data_utils
from abc import *
import random


class BaseDataloader(metaclass=ABCMeta):
    def __init__(self, args, mode, match_df, user_history_dict):
        self.args = args
        self.mode = mode
        self.match_df = match_df
        self.user_history_dict = user_history_dict
        
        seed = args.seed
        self.rng = random.Random(seed)
        self.sampler_rng = random.Random(seed) 
        
        self.args.PAD = 0
        self.args.MASK = 1
        self.args.CLS = 2
        self.args.UNK = 3    

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_dataloader(self):
        mode = self.mode
        batch_size = {'train':self.args.train_batch_size,
                      'val':self.args.val_batch_size,
                      'test':self.args.test_batch_size}[mode]
                      
        dataset = self._get_dataset()

        # use custom random sampler for reproducibility
        shuffle = False
        sampler = CustomRandomSampler(len(dataset), self.sampler_rng) if mode == 'train' else None
        drop_last = True if mode == 'train' else False
        dataloader = data_utils.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           pin_memory=True,
                                           num_workers=self.args.num_workers,
                                           drop_last=drop_last)
        return dataloader

    @abstractmethod
    def _get_dataset(self):
        pass
    
    
class CustomRandomSampler(data_utils.Sampler):
    def __init__(self, n, rng):
        super().__init__(data_source=[]) # dummy
        self.n = n
        self.rng = rng

    def __len__(self):
        return self.n

    def __iter__(self):
        indices = list(range(self.n))
        self.rng.shuffle(indices)
        return iter(indices)

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)