import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class HAM10000Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for the HAM10000 dataset.
    Where the labels are stored in the metadata csv file in the 'dx' column.
    and the image filenames are stored in the 'image_id' column.
    
    This updated version can handle multiple image directories.
    """
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataframe: Pandas DataFrame containing metadata
            image_dir: Path to image directory or list of paths to search
            transform: Optional transform to be applied to images
        """
        self.dataframe = dataframe
        # image_dir can be a single directory or a list of directories
        self.image_dir = image_dir
        self.transform = transform
        self.classes = sorted(dataframe['dx'].unique())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row['image_id']
        
        # Handle multiple directories
        if isinstance(self.image_dir, list):
            # Try to find the image in each directory
            image_path = None
            for dir_path in self.image_dir:
                potential_path = os.path.join(dir_path, image_id)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                raise FileNotFoundError(f"Image {image_id} not found in any provided directories")
        else:
            # Single directory
            image_path = os.path.join(self.image_dir, image_id)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            raise
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise
        
        label = self.classes.index(row['dx'])

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self, unique=True):
        """
        Returns either the unique classes or ALL labels.
        
        Args:
            unique: If True, return unique class names; otherwise, return all labels
        
        Returns:
            List of class names or all labels
        """
        if unique:
            return self.classes
        return self.dataframe['dx'].tolist()

    def get_class_weights(self):
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            numpy.ndarray: Class weights using balanced weighting
        """
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(self.classes),
            y=self.dataframe["dx"].to_numpy()
        )
        return weights