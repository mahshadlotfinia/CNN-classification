"""
Created on Sep 23, 2023.
data_provider.py

@author: Mahshad Lotfinia <lotfinia@wsa.rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""


import torch
from torchvision import transforms, datasets
import pdb
import os
import pandas as pd
import shutil
from torch.utils.data import dataloader, Dataset
import skimage
import numpy as np
from PIL import Image
from config.serde import read_config
import csv
from sklearn.model_selection import train_test_split

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15

class mydataset(Dataset):
    def __init__(self, data_input_path, CSV_input_path, valid_split_ration = 0.2,
                 transform = transforms.Compose(transforms.ToTensor()), seed = 42):

        self.data_input_path = data_input_path
        self.CSV_input_path = CSV_input_path
        self.transform = transform

        self.input_list = []
        self.sum_crack = 0
        self.sum_inactive = 0

        with open(CSV_input_path) as csv_file:
            reader = csv.DictReader (csv_file, delimiter = ';')
            for row in reader:
                self.input_list.append(row)

                if valid_split_ration == 0:
                    self.train_list = self.input_list
                else:
                    self.train_list, self.valid_list = train_test_split(
                        self.input_list, test_size = valid_split_ration, random_state = seed
                    )

                for row in self.train_list:
                    self.sum_crack += int [row('crack')]
                    self.sum_inactive += int[row('inactive')]


        pass




    def __len__(self):
        pass









    def __getitem__(self, item):
        pass





if __name__ == '__main__':




