import numpy as np
import tqdm

def read_large_array(path):
    try:
        blocksize = 512
        mmap = np.load(path, mmap_mode='r')
        array = np.empty_like(mmap)
        n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
        for b in tqdm.tqdm(range(n_blocks)):
            array[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
    finally:
        del mmap  # make sure file is closed again
    
    return array