import torch
import logging
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class EarlyStopping:
    # Early stopping to stop the training when the loss does not improve after *20* epochs.
    def __init__(self, model_name, patience=20, verbose=False, save_dir="models"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        self.model_path = os.path.join(save_dir, f"{model_name}.pth")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss


def train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, num_epochs,
        return_history=False
):
    # Initialize lists to track training progress
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"  Training loss: {avg_train_loss:.4f}")
        print(f"  Training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy = correct / total * 100
        val_accuracies.append(val_accuracy)

        logging.info(f"  Validation loss: {avg_val_loss:.4f}")
        logging.info(f"  Validation accuracy: {val_accuracy:.2f}%")
        print(f"  Validation loss: {avg_val_loss:.4f}")
        print(f"  Validation accuracy: {val_accuracy:.2f}%")

        # Scheduler step
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"  Learning rate: {current_lr}")
        print(f"  Learning rate: {current_lr}")

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logging.info("  Early stopping triggered!")
            print("  Early stopping triggered!")
            break

    if return_history:
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

    return val_accuracy


def test_model(model, test_loader, device, class_names, save_confusion_matrix=False):
    # Tests the model on the given test data loader.
    model.eval()
    true_labels = []
    pred_labels = []
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append true and pred labels to lists
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

            # Calc accuracy
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    # Overall accuracy
    test_accuracy = (correct / total) * 100

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Save confusion matrix
    if save_confusion_matrix:
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig("confusion_matrix.png")
        plt.close()

    # Classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)

    # Logging
    print("Test Accuracy: {:.2f}%".format(test_accuracy))
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    # Return results
    return {
        "accuracy": test_accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": {
            'report': report,
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }
    }


def load_model(model, checkpoint_path, device):
    # Load model weights from a checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model