import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class HAM10000Dataset(torch.utils.data.Dataset):
    # Custom dataset class for the HAM10000 dataset.
    # Where the labels are stored in the metadata csv file in the 'dx' column.
    # and the image filenames are stored in the 'image_id' column.
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.classes = sorted(dataframe['dx'].unique())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_id'])
        image = Image.open(image_path).convert("RGB")
        label = self.classes.index(row['dx'])

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self, unique=True):
        # Returns either the unique classes or ALL labels
        if unique:
            return self.classes
        return self.dataframe['dx'].tolist()

    def get_class_weights(self):
        # Calculate class weights for imbalanced datasets
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(self.classes),
            y=self.dataframe["dx"].to_numpy()
        )
        return weights


