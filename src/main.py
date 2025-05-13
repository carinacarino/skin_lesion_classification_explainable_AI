import logging
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
from dataloader import create_dataloaders
from models import get_model
from train_utils import train_model, EarlyStopping, test_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config


def main():
    # Setup logging for training process
    logging.basicConfig(
        filename="training.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting the training script.")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs(config["models_dir"], exist_ok=True)

    # Display model being trained
    model_name = config["model_name"]
    separator = "=" * 50
    model_banner = f"""
{separator}
    TRAINING MODEL: {model_name.upper()}
{separator}
"""
    print(model_banner)
    logging.info(f"Training model: {model_name}")

    # Prepare DataLoaders using our create_dataloaders function
    # Using both HAM10000 parts for training/validation and ISIC Test set for testing
    logging.info("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=config["metadata_path"],
        image_dirs=config["image_dirs"],
        test_metadata_path=config.get("test_metadata_path"),
        test_image_dir=config.get("test_image_dir"),
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=4,
        use_balanced_sampling=True  # Enable balanced batch sampling
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
        patience=10,
        verbose=True,
        save_dir=config["models_dir"]
    )

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Train the model
    logging.info(f"Starting training for {config['model_name']}...")
    train_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        num_epochs=config["num_epochs"],
        return_history=True  # Return history for plotting
    )

    # Store training history
    if isinstance(train_history, dict) and 'train_losses' in train_history:
        # New version of train_model that returns history
        train_losses = train_history['train_losses']
        val_losses = train_history['val_losses']
        val_accuracies = train_history['val_accuracies']

        # Plot training curves
        plt.figure(figsize=(15, 5))

        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{model_name} - Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        plt.title(f'{model_name} - Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_training_curves.png")
        plt.close()

    # Test the model immediately after training
    logging.info("Testing the model on the test set...")

    # Load the best model for testing
    best_model = get_model(config["model_name"], config["num_classes"], freeze_layers=False)
    best_model_path = os.path.join(config["models_dir"], f"{config['model_name']}.pth")
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)

    model.eval()
    class_names = train_loader.dataset.get_classes()

    # Get test results
    test_results = test_model(
        model=best_model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_confusion_matrix=False
    )

    # Compute confusion matrix
    conf_matrix = test_results["confusion_matrix"]
    logging.info("Confusion matrix computed.")

    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {config['model_name']}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"results/{config['model_name']}_confusion_matrix.png")
    plt.close()
    logging.info(f"Confusion matrix saved as results/{config['model_name']}_confusion_matrix.png")

    # Print classification report
    report = classification_report(
        test_results["classification_report"]["true_labels"],
        test_results["classification_report"]["pred_labels"],
        target_names=class_names
    )
    logging.info("\n" + report)
    print(report)

    # Save classification report to file
    with open(f"results/{config['model_name']}_classification_report.txt", "w") as f:
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Test Accuracy: {test_results['accuracy']:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    logging.info(f"Training and evaluation completed for {config['model_name']}")
    print(f"Training and evaluation completed for {config['model_name']}")


if __name__ == "__main__":
    main()