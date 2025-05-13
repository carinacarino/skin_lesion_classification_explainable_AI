import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from ham10000_dataset import HAM10000Dataset


def create_dataloaders(metadata_path, image_dirs, test_metadata_path=None, test_image_dir=None,
                       image_size=(224, 224), batch_size=32, num_workers=4, use_balanced_sampling=True):
    """
    Create dataloaders for training, validation, and testing.

    Args:
        metadata_path: Path to HAM10000 metadata CSV
        image_dirs: Dictionary with paths to image directories
        test_metadata_path: Path to ISIC test metadata CSV
        test_image_dir: Path to ISIC test images directory
        image_size: Target image size for resizing
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_balanced_sampling: If True, use balanced batch sampling for training

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for each split
    """
    # Load HAM10000 metadata
    df = pd.read_csv(metadata_path)

    # Ensure image_id has .jpg extension to match the actual filenames
    if not df['image_id'].str.contains('.jpg').any():
        df['image_id'] = df['image_id'] + '.jpg'

    # Split combined data into training and validation sets (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Load ISIC test set if provided
    if test_metadata_path and test_image_dir:
        test_df = pd.read_csv(test_metadata_path)
        # Ensure image_id has .jpg extension
        if not test_df['image_id'].str.contains('.jpg').any():
            test_df['image_id'] = test_df['image_id'] + '.jpg'
        print(f"Test samples: {len(test_df)}")
    else:
        # Fallback to using part of validation set for testing if no test set provided
        val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df['dx'], random_state=42)
        print(f"Test samples: {len(test_df)}")

    # Define transformations
    # Due to limited data, we apply multiple augmentations for training
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

    # Less transformations for validation and test sets
    transform_val_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    # For training and validation, we need to look in both part_1 and part_2 directories
    image_dirs_list = [image_dirs["part_1"], image_dirs["part_2"]]

    train_dataset = HAM10000Dataset(train_df, image_dirs_list, transform=transform_train)
    val_dataset = HAM10000Dataset(val_df, image_dirs_list, transform=transform_val_test)

    # For testing, use the dedicated ISIC test set if provided, otherwise use HAM10000 dirs
    if test_image_dir:
        test_dataset = HAM10000Dataset(test_df, test_image_dir, transform=transform_val_test)
    else:
        test_dataset = HAM10000Dataset(test_df, image_dirs_list, transform=transform_val_test)

    # Create DataLoaders
    if use_balanced_sampling:
        # Implement balanced batch sampling for the training set
        # Get all labels
        labels = train_df['dx'].tolist()

        # Get class frequencies
        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # Calculate weight for each sample (inverse of class frequency)
        weights = []
        for label in labels:
            weight = 1.0 / class_counts[label]
            weights.append(weight)

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        # Use the sampler with the DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        print("Using balanced batch sampling for training")

    else:
        # Regular random sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # For validation and test, we want to maintain the original distribution
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader