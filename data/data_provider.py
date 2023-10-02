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




class dataloader_:
    def __init__(self, data_dir):
        self.data_dir = '/home/mahshad/Documents/datasets/solar_cell_project/images'

    def __len__(self):
        return len(self.data_dir)

    def data_seperation(self, csv_path):
        df = pd.read_csv('/home/mahshad/Documents/datasets/solar_cell_project/master_list_with_splits.csv')
        for index, row in df.iterrows():
            image_name = row['filename']
            label_name = row['split']

            source_path = os.path.join(self.data_dir, image_name)
            if label_name == 'train':
                destination_path = os.path.join(self.data_dir, 'train', image_name)
            elif label_name == 'test':
                destination_path = os.path.join(self.data_dir, 'test', image_name)

            shutil.move(source_path, destination_path)


    def _transformation(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])#disired image size for ResNet 18
            ])

    def _loaddata(self):
        transform = self._transformation()

        train_dataset = datasets.ImageFolder (root = self.data_dir + 'train/', transform = transform)
        test_dataset = datasets.ImageFolder (root = self.data_dir + 'test/', transform = transform)

        train_loader = torch.utils.data.Dataloader (train_dataset, batch_size = 64, shuffle = True)
        test_loader = torch.utils.data.Dataloader (test_dataset, batch_size = 64, shuffle = False)

        return train_loader, test_loader

pdb.set_trace()
# data_handler = dataloader_('/home/mahshad/Documents/datasets/solar_cell_project/images')
# train_loader, _ = data_handler._loaddata()  # Get the train loader

# Iterate over the train loader to get one batch of data
# for images, labels in train_loader:
#     single_image = images[0]  # Get the first image in the batch
#     single_label = labels[0]  # Get the label corresponding to the first image
#     break  # Exit the loop after the first batch

# Now you can access and visualize the single image and its label





























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
