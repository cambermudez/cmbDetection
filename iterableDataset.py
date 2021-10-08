
# %%
"""
# Iterable Dataset
If we randomize the order of samples in the dataset itself, our dataset can simply iterate through samples.
"""

# %%
import h5py
import numpy as np
from torch.utils.data import Dataset,get_worker_info,DataLoader

# %%
from itertools import chain, islice, cycle

# %%
class simpleSlabDataset(Dataset):
    
    def __init__(self, data_file, batch_size=100, tp_percent=0.5, group=None):
        assert tp_percent<1
        
        self.data = h5py.File(data_file,'r')
        self.group = group   # train, valid, or test
        self.tp_len = self.data['/' + self.group + '/true_pos_slabs'].shape[0]
        self.fp_len = self.data['/' + self.group + '/false_pos_slabs'].shape[0]
        self.batch_size = batch_size
        self.tp_percent = tp_percent
        
        self.last_tp = 0
        self.last_fp = 0
        
        self.tp_perBatch = int(self.batch_size*tp_percent)
        self.fp_perBatch = int(self.batch_size-self.tp_perBatch)
                
    def __getitem__(self, inx):

        #Flip a coin for either FP or TP, still bounded by # of TP.
        if np.random.uniform() < 0.5:

            tp_ix = np.random.randint(0,self.tp_len)
            x = self.data['/' + self.group + '/true_pos_slabs'][tp_ix, 0, ...]
            x = np.nan_to_num(x,nan=np.nanmean(x))
            y = np.array(1)
        else:
            fp_ix = np.random.randint(0, self.fp_len)
            x = self.data['/' + self.group + '/false_pos_slabs'][fp_ix,0,...]
            x = np.nan_to_num(x,nan=np.nanmean(x))
            y = np.array(0)

        x=np.transpose(x,[3,0,1,2])
        y = y[...,np.newaxis]
#        print(x.shape,y.shape)
        return x.astype(np.float32), y.astype(np.float32)

#    def __iter__(self):
#        print('iter called')
#        return self.__getitem__(self.data)
 
    def __len__(self):
        #twice bc we sample for each TP and FP
        return (self.tp_len*2)
    
#    def tp_perBatch(self):
#        return self.tp_perBatch

