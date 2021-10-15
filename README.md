# cmbDetection
AI for CMB Detection

## Train.py

This works as the main.py script for training the network. Briefly, this handles data loading, training (model updating 
and backpropagation), and validation.  

Inputs (bash args)
-- epochs 
-- batch-size

Outputs (explicit)
1. Trained network for each epoch
2. Logs for training and validation loss

## Test.py

Similar to train.py,  but will evaluate the validation and testing set and save results as a mat file. 


## cmbNET.py
Network architecture

## iterableDataset.py
Generates callable dataset for pytorch

