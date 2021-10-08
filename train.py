import argparse
import pandas as pd
from tqdm import tqdm
import sys
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
for x,y in training_loader:
	break
print(x.shape)
print(y.shape)
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
    for x, y in tqdm(training_loader):

        # Forward pass
        y_hat = cmbNet(x.to(gpu_device))
        tr_loss =  criterion(y_hat.to(gpu_device),y.to(gpu_device))
        training_losses.append(tr_loss.detach().cpu().numpy().item())

        y_tr_epoch.extend(y.numpy())
        yhat_tr_epoch.extend(y_hat.detach().cpu().numpy())

        # Backpropagation
        opt.zero_grad()
        tr_loss.backward()
        opt.step()
#        break
	
    print("Starting Validation Set Evaluation... \n")
    # Validation
    with torch.no_grad():   # Freeze weights, no backprop -- MUCH faster computations

        cmbNet.eval()      # Remove dropout effects
        for x, y in tqdm(validation_loader):
            y_hat = cmbNet(x.to(gpu_device))
            val_loss = criterion(y_hat.to(gpu_device), y.to(gpu_device))
            validation_losses.append(val_loss.detach().cpu().numpy().item())

            y_val_epoch.extend(y.numpy())
            yhat_val_epoch.extend(y_hat.detach().cpu().numpy())
#            break
    
    # Display Cross-Entropy Loss for each epoch
    print(f"Epoch {epoch} : \n "
          f"Training Loss: {np.mean(training_losses):.2f} \n"
          f"Validation Loss: {np.mean(validation_losses):.2f} \n"
          f"\n"
          f"Training AUC {roc_auc_score(y_tr_epoch,yhat_tr_epoch):.2f}: \n"
          f"Validation AUC {roc_auc_score(y_val_epoch,yhat_val_epoch):.2f} \n")

    curve_csv.append([epoch,np.mean(training_losses),np.mean(validation_losses)])

    # Save model for each epoch
    net_fname = '/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/trained_cmbNet_epoch' + str(epoch) + '.h5'
    torch.save(cmbNet.state_dict(), net_fname)


print("Done Training!")
df = pd.DataFrame(curve_csv,columns=['Epoch','Training Loss','Validation Loss'])
df.to_csv('/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/training_curve.csv')



## TO DO:
# Testing -- different script maybe (load model and evaluate)




