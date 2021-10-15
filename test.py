import argparse
import pandas as pd
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
from scipy.io import savemat
import os
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
parser.add_argument('--batch-size', required=False, default = 64, help='Number of epochs to train', type=int)
parser.add_argument('--experiment-date', required=True, help='Experiment Date under ./workdir/', type=str)
parser.add_argument('--workdir', required=False,default='/mnt/j6/m252055/20211004_cmbDetection/20211004_preprocessed/', type=str)
parser.add_argument('--load-epoch', required=True, help='Model Epoch to load', type=str)
args = parser.parse_args()

## Set up data loader
data_file = os.path.join(args.workdir,'20211008_cmbOrderedDataset.hdf5')

print("Making Data Loaders")
validation_ds = simpleSlabDataset(data_file,group='valid')
test_ds       = simpleSlabDataset(data_file,group='test')

validation_loader = DataLoader(validation_ds, batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=1)
test_loader = DataLoader(test_ds, batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=1)

cmbNet = cmbNet()
cmbNet.load_state_dict(torch.load(os.path.join(args.workdir,args.experiment_date,'trained_cmbNet_epoch'+args.load_epoch+'.h5')))

cmbNet.to(gpu_device)
criterion = torch.nn.BCELoss()

print("Starting Validation Set Evaluation... \n")
# Validation
with torch.no_grad():   # Freeze weights, no backprop -- MUCH faster computations

    validation_losses = []
    testing_losses = []

    y_val_epoch = []
    yhat_val_epoch = []

    y_te_epoch = []
    yhat_te_epoch = []

    cmbNet.eval()      # Remove dropout effects
    for x, y in tqdm(validation_loader):
        y_hat = cmbNet(x.to(gpu_device))
        val_loss = criterion(y_hat.to(gpu_device), y.to(gpu_device))
        validation_losses.append(val_loss.detach().cpu().numpy().item())

        y_val_epoch.extend(y.numpy())
        yhat_val_epoch.extend(y_hat.detach().cpu().numpy())
#        break

    print("Starting Testing Set Evaluation ... \n")
    for x, y in tqdm(test_loader):
        y_hat = cmbNet(x.to(gpu_device))
        test_loss = criterion(y_hat.to(gpu_device), y.to(gpu_device))
        testing_losses.append(val_loss.detach().cpu().numpy().item())

        y_te_epoch.extend(y.numpy())
        yhat_te_epoch.extend(y_hat.detach().cpu().numpy())
#        break


print(f"Validation Loss: {np.mean(validation_losses):.2f} \n"
      f"Testing Loss: {np.mean(testing_losses):.2f} \n"
      f"\n"
      f"Validation AUC {roc_auc_score(y_val_epoch,yhat_val_epoch):.2f}: \n"
      f"Testing AUC {roc_auc_score(y_te_epoch,yhat_te_epoch):.2f} \n")


savemat(os.path.join(args.workdir,args.experiment_date,'final_predictions.mat'),
        mdict={"y_val":y_val_epoch,"yhat_val":yhat_val_epoch,
               "y_test":y_te_epoch,"yhat_test":yhat_te_epoch})


