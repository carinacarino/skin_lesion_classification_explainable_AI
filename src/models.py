from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
    DenseNet121_Weights,
    ViT_B_16_Weights,

)
from torch import nn

# Reference:
# https://python.plainenglish.io/how-to-freeze-model-weights-in-pytorch-for-transfer-learning-step-by-step-tutorial-a533a58051ef
def get_model(model_name, num_classes, freeze_layers=True):
    # We are using the torchvision library to get the models.
    # The models are pre-trained on the ImageNet dataset.
    # We can also freeze the layers of the mode.
    # Then change the last layer to output the number of classes we have.
    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_layers:
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if freeze_layers:
            for param in model.features[:10].parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_layers:
            for param in model.features[:5].parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        if freeze_layers:
            for param in model.features[:6].parameters():
                param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_layers:
            for param in model.encoder.parameters():
                param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model
