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




















































































































































































###################################################################################################################################################33
# class dataloader_:
#     def __init__(self, data_dir, csv_path):
#         self.data_dir = data_dir
#         self.csv_path = csv_path
#
#     def separate_data(self, train_folder, test_folder):
#         df = pd.read_csv(self.csv_path)
#
#         # Create train and test folders if they don't exist
#         os.makedirs(train_folder, exist_ok=True)
#         os.makedirs(test_folder, exist_ok=True)
#
#         for index, row in df.iterrows():
#             image_name = row['filename'].replace('images/', '')  # Removing 'images/' prefix
#             label_name = row['split']
#
#             source_path = os.path.join(self.data_dir, image_name)
#             if label_name == 'train':
#                 destination_path = os.path.join(train_folder, image_name)
#             elif label_name == 'test':
#                 destination_path = os.path.join(test_folder, image_name)
#
#             if os.path.exists(source_path):
#                 os.makedirs(os.path.dirname(destination_path), exist_ok=True)
#                 shutil.move(source_path, destination_path)
#             else:
#                 print(f"Image not found: {source_path}")
#         return destination_path
#
#
#     def _transformation(self, image):
#         pdb.set_trace()
#         transformed = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         pdb.set_trace()
#
#         transformed_image = transformed(image)
#         return transformed_image
#
#
#
#     def _load_data_(self, transformed_train, transformed_test):
#         result_transformed_train = self._transformation(transformed_train)
#         result_transformed_test = self._transformation(transformed_test)
#
#         train_dataset = datasets.ImageFolder (root = os.path.join(self.data_dir, 'train_folder/'), transform = result_transformed_train)
#         test_dataset = datasets.ImageFolder (root = os.path.join(self.data_dir, 'test_folder/'), transform = result_transformed_test)
#
#         train_loader = torch.utils.data.DataLoader (train_dataset, batch_size = 64, shuffle = True)
#         test_loader = torch.utils.data.DataLoader (test_dataset, batch_size = 64, shuffle = False)
#
#         return train_loader, test_loader
#
#
#
#
#
#
# if __name__ == '__main__':
#
#
#     data_dir = '/home/mahshad/Documents/datasets/solar_cell_project/images'
#     csv_path = '/home/mahshad/Documents/datasets/solar_cell_project/master_list_with_splits.csv'
#     train_folder = '/home/mahshad/Documents/datasets/solar_cell_project/images/train_folder'
#     test_folder = '/home/mahshad/Documents/datasets/solar_cell_project/images/test_folder'
#     data_handler = dataloader_(data_dir,csv_path)
#     result1 = data_handler.separate_data(train_folder, test_folder)
#
#     train_images = [os.path.join(train_folder, file) for file in os.listdir(train_folder)]
#
#     for image in train_images:
#         loaded_image = skimage.io.imread(image)
#         pdb.set_trace()
#
#     transformed_train_data = [data_handler._transformation(skimage.io.imread(image)) for image in train_images]
#
#     test_images = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
#     transformed_test_data = [data_handler._transformation(skimage.io.imread(image)) for image in test_images]
#
#     train_loader, test_loader = data_handler._load_data_(transformed_train_data, transformed_test_data)    #
#     #
#     #
#     # data_handler = dataloader_('/home/mahshad/Documents/datasets/solar_cell_project/images')
#     # train_loader, _ = data_handler._loaddata()  # Get the train loader
#
#     # Iterate over the train