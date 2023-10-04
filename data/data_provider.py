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
from Train_Test_Valid import Mode
from skimage.io import imread
from skimage.color import gray2rgb



train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15

class mydataset(Dataset):
    def __init__(self, data_input_path, CSV_input_path, valid_split_ration,
                 transform = transforms.Compose(transforms.ToTensor()), mode = Mode.Train, seed = 42):

        self.data_input_path = data_input_path
        self.CSV_input_path = CSV_input_path
        self.transform = transform
        self.valid_split_ration = valid_split_ration

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
                    self.sum_crack += int(row['crack'])
                    self.sum_inactive += int(row['inactive'])



        pass




    def __len__(self):
        if self.input_list == self.train_list:
            return len(self.train_list)
        else:
            return len(self.valid_list)




    def __getitem__(self, idx):
        self.output_list = []

        if self.mode == Mode.Train:
            self.output_list = self.train_list
        elif self.mode == Mode.Valid:
            self.output_list = self.valid_list

        label = np.zeros((2), dtype = int)
        label[0] = int (self.output_list[idx]['crack'])
        label[1] = int (self.output_list [idx] ['inactive'])

        image = imread(os.path.join(self.data_input_path), self.output_list[idx]['filename'])
        image = gray2rgb(image)
        image = self.transform(image)
        label = torch.from_numpy(label)
        return image, label





if __name__ == '__main__':




