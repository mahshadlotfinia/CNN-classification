import os
import shutil
import pandas as pd

def separate_images(csv_path, source_folder, train_folder, test_folder):
    df = pd.read_csv(csv_path)

    # Create train and test folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for index, row in df.iterrows():
        image_name = row['filename'].replace('images/', '')  # Removing 'images/' prefix
        label_name = row['split']

        source_path = os.path.join(source_folder, image_name)
        if label_name == 'train':
            destination_path = os.path.join(train_folder, image_name)
        elif label_name == 'test':
            destination_path = os.path.join(test_folder, image_name)

        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.move(source_path, destination_path)
        else:
            print(f"Image not found: {source_path}")

# Usage example:
csv_path = '/home/mahshad/Documents/datasets/solar_cell_project/master_list_with_splits.csv'
source_folder = '/home/mahshad/Documents/datasets/solar_cell_project/images'
train_folder = '/home/mahshad/Documents/datasets/solar_cell_project/images/train_folder'
test_folder = '/home/mahshad/Documents/datasets/solar_cell_project/images/test_folder'

separate_images(csv_path, source_folder, train_folder, test_folder)
