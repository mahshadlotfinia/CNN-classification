"""
Created on Sep 23, 2023.
data_provider.py

@author: Mahshad Lotfinia <lotfinia@wsa.rwth-aachen.de>
https://github.com/mahshadlotfinia/
"""


from torch.utils.data import Dataset
import torch
import csv
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from Train_Test_Valid import Mode
# from configs.serde import read_config
import os.path
import pdb
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15


class mydataset(Dataset):
    def __init__(self, data_input_path, CSV_input_path, valid_split_ration, mode = 'train',
                 transform = transforms.Compose(transforms.ToTensor()), seed = 42):

        self.data_input_path = data_input_path
        self.CSV_input_path = CSV_input_path
        self.transform = transform
        self.valid_split_ration = valid_split_ration
        self.mode = mode

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





    def __len__(self):
        if self.input_list == self.train_list:
            return len(self.train_list)
        else:
            return len(self.valid_list)




    def __getitem__(self, idx):
        self.output_list = []

        if self.mode == 'train':
            self.output_list = self.train_list
        elif self.mode == 'valid':
            self.output_list = self.valid_list

        label = np.zeros((2), dtype = int)
        label[0] = int (self.output_list[idx]['crack'])
        label[1] = int (self.output_list [idx] ['inactive'])

        image = imread(os.path.join(self.data_input_path), self.output_list[idx]['filename'])
        image = gray2rgb(image)
        image = self.transform(image)
        label = torch.from_numpy(label)
        return image, label


    def pos_weight(self):
        W_crack = torch.tensor((len(self.train_list) - self.sum_crack) / (self.sum_crack + epsilon))
        W_inactive = torch.tensor((len(self.train_list) - self.sum_inactive) / (self.sum_inactive + epsilon))
        output_tensor = torch.zeros((2))
        output_tensor[0] = W_crack
        output_tensor[1] = W_inactive

        return output_tensor

def get_train_dataset(data_input_path, valid_split_ration):
    trans = transforms.Compose ([transforms.ToPILImage(), transforms.RandomVerticalFlip(p = 0.5),
                                transforms.RandomHorizontalFlip(p = 0.5), transforms.ToTensor(),
                                transforms.Normalize(train_mean, train_std)])
    return mydataset (data_input_path = data_input_path, valid_split_ration = valid_split_ration, transform = trans, mode = 'train')

def get_validation_dataset(data_input_path, valid_split_ration):
    trans = transforms.Compose ([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(train_mean, train_std)])
    return mydataset (data_input_path = data_input_path, valid_split_ration = valid_split_ration, transform = trans, mode = 'Valid')


if __name__ == '__main__':
    DATA_PATH = '/home/mahshad/Documents/datasets/solar_cell_project/images'
    train_dataset = get_train_dataset(DATA_PATH, valid_split_ration = 0.2)
    valid_dataset = get_validation_dataset(DATA_PATH, valid_split_ration = 0.2)
    # pdb.set_trace()

