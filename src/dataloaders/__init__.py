from .base import BaseDataloader
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseDataloader)


DATALOADERS = {c.code():c
               for c in all_subclasses(BaseDataloader)
               if c.code() is not None}


def init_dataloader(args, match_df, user_history_dict):
    train_dataloader = DATALOADERS[args.train_dataloader_type](args, 'train', match_df['train'], user_history_dict)
    val_dataloader = DATALOADERS[args.val_dataloader_type](args, 'val', match_df['val'], user_history_dict)
    test_dataloader = DATALOADERS[args.test_dataloader_type](args, 'test', match_df['test'], user_history_dict)

    train_dataloader = train_dataloader.get_dataloader()
    val_dataloader = val_dataloader.get_dataloader()
    test_dataloader = test_dataloader.get_dataloader()
    
    return train_dataloader, val_dataloader, test_dataloader