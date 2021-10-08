
# %%
"""
# Iterable Dataset
If we randomize the order of samples in the dataset itself, our dataset can simply iterate through samples.
"""

# %%
import h5py
import numpy as np
from torch.utils.data import IterableDataset,get_worker_info,DataLoader

# %%
from itertools import chain, islice, cycle

# %%
data_source = '/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/20210827_cmbOrderedDataset.hdf5'
data_file = h5py.File(data_source,'r')

# %%
class simpleSlabDataset(IterableDataset):
    
    def __init__(self, data_file, batch_size=100, tp_percent=0.5, group=None):
        assert tp_percent<1
        
        self.data = data_file
        self.group = group   # train, valid, or test
        self.tp_len = data_file['/' + self.group + '/true_pos_slabs'].shape[0]
        self.fp_len = data_file['/' + self.group + '/false_pos_slabs'].shape[0]
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
            y = 1
        else:
            fp_ix = np.random.randint(0, self.fp_len)
            x = self.data['/' + self.group + '/false_pos_slabs'][fp_ix,0,...]
            y = 0

        print(x.shape,y.shape)
        return (x, y)


    def __iter__(self):
        print('iter called')
        return self.get_data(self.data)
    
    def __len__(self):
        #twice bc we sample for each TP and FP
        return 2*self.tp_len
    
    def tp_perBatch(self):
        return self.tp_perBatch

