import logging
import os
import torch
from datetime import datetime
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from dataloader import create_dataloaders
from models import get_model
from train_utils import train_model, EarlyStopping, test_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
import pandas as pd
import numpy as np


def main():
    # Logg the training process
    logging.basicConfig(
        filename="training.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting the training script.")

    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Prepare DataLoaders using our create_dataloaders function
    # You can edit the parameters in the config file
    logging.info("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=config["metadata_path"],
        image_dirs=config["image_dirs"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=4,
    )

    # Sanity check
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    logging.info(f"Testing samples: {len(test_loader.dataset)}")

    # Load chosen model from model.py, note that we are freezing early layers
    logging.info(f"Loading model: {config['model_name']}...")
    model = get_model(config["model_name"], config["num_classes"], freeze_layers=True)
    model = model.to(device)
    logging.info("Model loaded successfully.")

    # Define classweights (bc classes are unbalanced), loss function, optimizer, and scheduler
    class_weights = torch.tensor(
        train_loader.dataset.get_class_weights(), dtype=torch.float, device=device
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # Learning rate decays after 5 epochs without improvement
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    logging.info("Loss function, optimizer, and scheduler defined.")

    # Early stopping to avoid overfitting. Saves the model with the best validation loss
    # This is part of the train_utils.py script
    # Patience is zero, meaning that the model stops training when the validation loss stops decreasing by 20 epochs
    early_stopping = EarlyStopping(
        model_name=config["model_name"],
        patience=20,
        verbose=True,
        save_dir="models"
    )

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        num_epochs=config["num_epochs"],
    )

    # Test the model immedietly after training
    logging.info("Testing the model on the test set...")
    model.eval()
    true_labels = []
    pred_labels = []
    class_names = train_loader.dataset.get_classes()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    logging.info("Confusion matrix computed.")

    # Display confusion matrix
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix saved as confusion_matrix.png.")

    # Print classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    logging.info("\n" + report)
    print(report)


if __name__ == "__main__":
    main()
