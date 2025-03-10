import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from ham10000_dataset import HAM10000Dataset


def create_dataloaders(metadata_path, image_dirs, image_size, batch_size, num_workers=4):
    # Function to create dataloaders for train, val and test.
    df = pd.read_csv(metadata_path)

    # Just to make sure the image_id has the correct extension to find the correct image
    df['image_id'] = df['image_id'] + ".jpg"

    # Since the dataset is divided in to two parts
    # We use part 1 for training
    # Part 2 is divided into validation and testing
    part_1_dir = image_dirs["part_1"]
    part_2_dir = image_dirs["part_2"]

    df_part_1 = df[df['image_id'].isin(os.listdir(part_1_dir))]
    df_part_2 = df[df['image_id'].isin(os.listdir(part_2_dir))]

    # Verify splits
    if len(df_part_1) == 0 or len(df_part_2) == 0:
        raise ValueError("No matching files found")

    # Split part_2 into validation and testing
    # Note to self: It might have been better to split the data to atleast 75% training
    # and divide the remaining 35% into validation and testing
    df_val, df_test = train_test_split(df_part_2, test_size=0.5, stratify=df_part_2['dx'], random_state=42)

    print(f"Training samples: {len(df_part_1)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Testing samples: {len(df_test)}")

    # Define transformations
    # Due to limited data, we apply all the augmentations that I can think of
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = HAM10000Dataset(df_part_1, part_1_dir, transform=transform_train)
    val_dataset = HAM10000Dataset(df_val, part_2_dir, transform=transform_val_test)
    test_dataset = HAM10000Dataset(df_test, part_2_dir, transform=transform_val_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
