import argparse
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np

import torch
from torch.utils.data import DataLoader

from cmbNet import cmbNet
from iterableDataset import simpleSlabDataset

## CUDA Setup

use_cuda = torch.cuda.is_available()
gpu_device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

## To Do: Add parse args if necessary
parser = argparse.ArgumentParser(description='Run cmbNet detection of TP vs FP')
parser.add_argument('--epochs', required=False, default = 10, help='Number of epochs to train', type=int)
parser.add_argument('--batch-size', required=False, default = 100, help='Number of epochs to train', type=int)
args = parser.parse_args()

## Set up data loader
data_file = '/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/20210827_cmbOrderedDataset.hdf5'

print("Making Data Loaders")
train_ds      = simpleSlabDataset(data_file,group='train')
validation_ds = simpleSlabDataset(data_file,group='valid')
#test_ds       = simpleSlabDataset(data_file,group='test')

train_ds.__getitem__(0)

training_loader = DataLoader(train_ds, batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=1)
validation_loader = DataLoader(validation_ds, batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=1)
#test_loader = DataLoader(test_ds, batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=8)
cmbNet = cmbNet()
cmbNet.to(gpu_device)
criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(cmbNet.parameters(), lr=0.001)

# Instantiate the output csv
curve_csv = []

print("Starting Training...")
for epoch in range(args.epochs):

    training_losses   = []
    validation_losses = []

    y_tr_epoch = []
    yhat_tr_epoch = []

    y_val_epoch = []
    yhat_val_epoch = []

    # Training
    cmbNet.train()
    c=0
    for x, y in training_loader:

        # Forward pass
        y_hat = cmbNet(x.to(gpu_device))
        tr_loss =  criterion(y_hat.to(gpu_device),y.to(gpu_device))
        training_losses.append(tr_loss)

        y_tr_epoch.append(y)
        yhat_tr_epoch.append(y_hat)

        # Backpropagation
        opt.zero_grad()
        tr_loss.backward()
        opt.step()
	
        c += 1
        print(c)

    # Validation
    with torch.nograd():   # Freeze weights, no backprop -- MUCH faster computations

        cmbNet.eval()      # Remove dropout effects
        for x, y in validation_loader:
            y_hat = cmbNet(x.to(gpu_device))
            val_loss = criterion(y_hat.to(gpu_device), y.to(gpu_device))
            validation_losses.append(val_loss)

            y_val_epoch.append(y)
            yhat_val_epoch.append(y_hat)

    # Display Cross-Entropy Loss for each epoch
    print('Epoch %s : \n '
          'Training Loss: %f \n'
          'Validation Loss: %f \n'
          '\n'
          'Training AUC: \n'
          'Validation AUC \n',
          str(epoch), np.mean(training_losses),np.mean(validation_losses),roc_auc_score(y_tr_epoch,yhat_tr_epoch),roc_auc_score(y_val_epoch,yhat_val_epoch))

    curve_csv.append([epoch,np.mean(training_losses),np.mean(validation_losses)])
    print(curve_csv)

    # Save model for each epoch
    net_fname = '/mnt/j6/m252055/20210104_cmbDetection/20211004_preprocessed/trained_cmbNet_epoch' + str(epoch) + '.h5'
    torch.save(cmbNet.state_dict(), net_fname)


print("Done Training!")
df = pd.Dataframe(curve_csv,columns=['Epoch','Training Loss','Validation Loss'])
df.to_csv('/mnt/j6/m252055/20210104_cmbDetection/20211004_preprocessed/training_curve.csv')



## TO DO:
# Testing -- different script maybe (load model and evaluate)




