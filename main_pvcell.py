"""
Created on Sep 23, 2023.
main_pvcell.py

@author: Mahshad Lotfinia <lotfinia@wsa.rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import timm
import numpy as np
from sklearn import metrics

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_pvcell import Training
from Prediction_pvcell import Prediction
from data.data_provider import dataloader

import warnings
warnings.filterwarnings('ignore')




def main_train(global_config_path="/home/mahshad/Documents/Repositories/CNN-classification/config/config.yaml"):
    pass


def main_test(global_config_path="/home/mahshad/Documents/Repositories/CNN-classification/config/config.yaml"):
    pass





if __name__ == '__main__':
    main_train(global_config_path="/home/mahshad/Documents/Repositories/CNN-classification/config/config.yaml")
