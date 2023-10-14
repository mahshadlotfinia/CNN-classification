"""
Created on Sep 23, 2023.
data_provider.py

@author: Mahshad Lotfinia <lotfinia@wsa.rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""

import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import pdb
import pandas as pd
# import cv2
from torch import optim
from sklearn.datasets import make_regression

############################################################################################################################################
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        # this is for using GPU instead of CPU (data.to (device))
# pdb.set_trace()                                                                # this code use for debugging


############################################################################################################################################


#############################################################################################################################################
# linear regression

def dataprovider():
    n_features = 1
    n_samples = 100

    x , y = make_regression (
        n_samples = n_samples,
        n_features = n_features,
        noise = 10,
    )
    fix, ax = plt.subplots()
    ax.plot (x, y, '.')
    plt.show()
    x = torch.from_numpy(x).float()  # transfer from numpy to pytorch
    y = torch.from_numpy(y.reshape((n_samples, n_features))).float()



class LinReg (nn.Module):
    def __init__ (self, input_dim):
        super().__init__()
        self.beta = nn.Linear(input_dim, 1)

    def forward (self, x):
        return self.beta(x)


# transfer data to GPU
def movetoGPU():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# every deep learning code has model, optimizer and loss function
model = LinReg(n_features).to(device)
optimizer = optim.SGD (model.parameters(), lr = 0.001)
criterion = nn.MSELoss()
x, y = x.to(device), y.to(device)



# Define a function for training
def train(model, optimizer, criterion, x, y, device):
    model.train()
    optimizer.zero_grad()
    y_ = model(x)
    loss = criterion(y_, y)
    loss.backward()
    optimizer.step()
    return loss.item()


# Define a function for evaluation
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_ = model(x)
    return y_


# Continue training in a loop
num_epochs = 200  # Adjust as needed
best_loss = float('inf')  # Initialize with a high value

for epoch in range(num_epochs):
    current_loss = train(model, optimizer, criterion, x, y, device)

    if current_loss < best_loss:
        best_loss = current_loss
        best_model_state = model.state_dict()  # Save the model state

    if epoch % 10 == 0:  # Print progress every 10 epochs
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {current_loss:.4f}')

# Load the best model state
model.load_state_dict(best_model_state)

# Evaluate and visualize the data
y_ = evaluate(model, x, y)

fig, ax = plt.subplots()
ax.plot(x.cpu().numpy(), y_.cpu().numpy(), '.', label='pred')
ax.plot(x.cpu().numpy(), y.cpu().numpy(), '.', label='data')
ax.set_title(f"MSE: {best_loss:0.1f}")
ax.legend()
plt.show()

############################################################################################################################################


import skimage as ski



# epsilon = 1e-15




# class dataloader(Dataset):
#     """
#     This is the pipeline based on Pytorch's Dataset and Dataloader
#     """
#     def __init__(self, cfg_path, ):
#
#
#     def __len__(self):
#         """Returns the length of the dataset"""
#         pass
#         return
#
#
#     def __getitem__(self, idx):
#
#
#         return



























# if __name__ == '__main__':
