# %%
#%config Completer.use_jedi = False

# %%
import pandas as pd
import numpy as np
import h5py

# %%
def cmbMakeDataset(in_cmbTable,out_filePath):
    # First, shrink table down to unique scans
    tp_size = (0,200,3,33,33,5)
    fp_size = (0,2,3,33,33,5)

    unique_scansTable = in_cmbTable.drop_duplicates(subset='seriesuid')

    with h5py.File(out_filePath,'w') as hf:
#             del hf['/train/true_pos_slabs']
#             del hf['/train/false_pos_slabs']
#             del hf['/valid/true_pos_slabs']
#             del hf['/valid/false_pos_slabs']


        hf.create_dataset('/train/true_pos_slabs',tp_size,maxshape=(None,200,3,33,33,5),
                         dtype='f4',chunks=(10**4,2,3,33,33,5))
        hf.create_dataset('/train/false_pos_slabs',fp_size,maxshape=(None,2,3,33,33,5),
                         dtype='f4',chunks=(10**4,2,3,33,33,5))

        hf.create_dataset('/valid/true_pos_slabs',tp_size,maxshape=(None,200,3,33,33,5),
                         dtype='f4',chunks=(10**4,2,3,33,33,5))
        hf.create_dataset('/valid/false_pos_slabs',fp_size,maxshape=(None,2,3,33,33,5),
                         dtype='f4',chunks=(10**4,2,3,33,33,5))
        
        hf.create_dataset('/test/true_pos_slabs',tp_size,maxshape=(None,200,3,33,33,5),
                         dtype='f4',chunks=(10**4,2,3,33,33,5))
        hf.create_dataset('/test/false_pos_slabs',fp_size,maxshape=(None,2,3,33,33,5),
                         dtype='f4',chunks=(10**4,2,3,33,33,5))

        for ii,h5_path in enumerate(unique_scansTable['cmbCandidates_hdf5']):
            try:
                if unique_scansTable['tvtLabel'].iloc[ii]=='Train':
                    group = '/train'
                elif unique_scansTable['tvtLabel'].iloc[ii]=='Valid':
                    group = '/valid'
                elif unique_scansTable['tvtLabel'].iloc[ii]=='Test':
                    group = '/test'

                hdf5_file = h5py.File(h5_path,'r')
                tp_dset = np.array(hdf5_file['/data/TPblocks'])
                fp_dset = np.array(hdf5_file['/data/FPblocks'])

                if not tp_dset.size==1:
                    hf[group + '/true_pos_slabs'].resize(hf[group + '/true_pos_slabs'].shape[0] + tp_dset.shape[0],axis=0)
                    hf[group + '/true_pos_slabs'][-tp_dset.shape[0]:,:,:,:,:,:] = tp_dset;

                hf[group + '/false_pos_slabs'].resize(hf[group + '/false_pos_slabs'].shape[0] + fp_dset.shape[0],axis=0)
                hf[group + '/false_pos_slabs'][-fp_dset.shape[0]:,:,:,:,:,:] = fp_dset;
                
            except:
                # This is here because some rows in the table are wrong/corrupted. Maybe add a counter later to see which ones
                continue
            if ii > 10:
                break
    pass

# %%
# cmbTable = pd.read_csv('/mnt/j6/m167350/20210810_cmbDetection/')
cmbTable = pd.read_csv('/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/20210826_cmbProcessedTableForPython.csv',error_bad_lines=False)

# %%
# out_filePath = '/mnt/j6/m167350/20210810_cmbDetection/'
out_filePath = '/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/20210827_cmbOrderedDataset.hdf5'

# %%
print('Making Dataset...')
cmbMakeDataset(cmbTable,out_filePath)
print('Done!')
# %%
cmbTable.columns

# %%
f = h5py.File(out_filePath)
print(f['train'].keys())

# %%
print(f['/train/true_pos_slabs'].shape)
print(f['/valid/true_pos_slabs'].shape)
print(f['/test/true_pos_slabs'].shape)
