import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import IntegratedGradients, Saliency, GuidedBackprop, GuidedGradCam, Occlusion, visualization
src_path = os.path.abspath("../src")
if src_path not in sys.path:
    sys.path.append(src_path)
from models import get_model
from train_utils import load_model

# References:
# https://medium.com/pytorch/introduction-to-captum-a-model-interpretability-library-for-pytorch-d236592d8afa
# https://github.com/pytorch/captum
# https://pytorch.org/tutorials/beginner/introyt/captumyt.html
# https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html

# Function to analyze a single model
def analyze_model(model_name, image_path, output_dir, target_class=None):
    # Analyze one model with multiple attribution methods

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(model_name, num_classes=7, freeze_layers=False)
    checkpoint_path = f"../models/{model_name}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    # Load and process im
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Determine class
    if target_class is None:
        with torch.no_grad():
            target_class = model(input_tensor).argmax(dim=1).item()

    # Attribution methods
    methods = {
        'Integrated Gradients': IntegratedGradients(model),
        'Saliency': Saliency(model),
        'Guided Backprop': GuidedBackprop(model),
        'Occlusion': Occlusion(model)
    }

    # Add GuidedGradCam for CNNs (non-ViT models)
    if model_name != "vit_b_16":
        last_conv = None
        if hasattr(model, 'features'):  # For vgg, densenet, efficientnet
            last_conv = model.features[-1]
        elif hasattr(model, 'layer4'):  # For resnet
            last_conv = model.layer4[-1]
        if last_conv:
            methods['Guided GradCam'] = GuidedGradCam(model, last_conv)

    # Generate visualizations for each method
    n_methods = len(methods)
    plt.figure(figsize=(5 * n_methods, 4))
    for idx, (method_name, method) in enumerate(methods.items(), 1):
        plt.subplot(1, n_methods, idx)
        if method_name == 'Occlusion':
            attributions = method.attribute(
                input_tensor,
                target=target_class,
                strides=(3, 8, 8),
                sliding_window_shapes=(3, 15, 15),
                baselines=0
            )
        else:
            attributions = method.attribute(input_tensor, target=target_class)

        # Visualize attributions
        _ = visualization.visualize_image_attr(
            np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            title=f"{method_name}",
            plt_fig_axis=(plt.gcf(), plt.gca()),
            use_pyplot=False,
            cmap='seismic'
        )

    plt.suptitle(f"Model: {model_name} - Attribution Methods", y=1.05, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_attributions.png"), bbox_inches='tight', dpi=300,
                pad_inches=0.2)
    plt.close()



def analyze_all_models(models, image_path, output_dir):
    # Function to analyze multiple models and aggregate results to one figure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    for model_name in models:
        print(f"Processing {model_name}...")
        analyze_model(model_name, image_path, output_dir)


if __name__ == "__main__":
    models = ["resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b_16"]
    image_path = "../HAM10000/HAM10000_images_part_1/ISIC_0024308.jpg"
    output_dir = "../xai_results"
    analyze_all_models(models, image_path, output_dir)
