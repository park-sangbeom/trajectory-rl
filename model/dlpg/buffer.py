import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.utils.sampler import kdpp, ikdpp

class BufferClass:
    def __init__(self, xdim=2, cdim=5*2, buffer_limit=1000, device="cpu"):
        self.x      = np.zeros([buffer_limit, xdim], dtype=np.float32)
        self.c      = np.zeros([buffer_limit, cdim], dtype=np.float32)
        self.reward = np.zeros([buffer_limit], dtype=np.float32)
        self.ptr, self.size, self.buffer_limit   =   0, 0, buffer_limit 
        self.device = device 

    def store(self, x, c, reward):
        self.x[self.ptr]      = x 
        self.c[self.ptr]      = c 
        self.reward[self.ptr] = reward
        self.ptr              = (self.ptr+1)%self.buffer_limit
        self.size             = min(self.size+1, self.buffer_limit) # number of instance stored

    def sample_batch(self, sample_method="random", batch_size=128):    
        if sample_method == "kdpp":
            _, idxs = kdpp(xs_total=self.x, n_select=batch_size)
        elif sample_method == "ikdpp":
            _, idxs = ikdpp(xs_total=self.x, qs_total=None, n_trunc=30, n_select=batch_size) 
        elif sample_method == "ikqdpp":
            _, idxs = ikdpp(xs_total=self.x, qs_total=self.reward, n_trunc=30, n_select=batch_size)
        else: # Random 
            idxs = np.random.permutation(self.size)[:batch_size]
        batch = dict(
        x        = torch.tensor(self.x[idxs]).to(self.device), 
        c        = torch.tensor(self.c[idxs]).to(self.device), 
        reward   = torch.tensor(self.reward[idxs]).to(self.device)
        )
        return batch
    