config = {
    "model_name": "vit_b_16",  # Options: resnet18, vgg16, densenet121, efficientnet_b0, "vit_b_16"
    "num_classes": 7,          # Number of classes
    "batch_size": 64,          # Max batch size my GPU can handle
    "num_epochs": 100,
    "learning_rate": 0.0001,
    "image_size": (224, 224), # Resize the images to 224x224, this is the original im size the models were trained on
    "image_dirs": { # The images were stored in two separate folders, we need to combine them
        "part_1": "./HAM10000/HAM10000_images_part_1",
        "part_2": "./HAM10000/HAM10000_images_part_2"
    },
    "metadata_path": "./HAM10000/HAM10000_metadata.csv", # Metadata, where the true labels are stored
    "models_dir": "./models"
}
