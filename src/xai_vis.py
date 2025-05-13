import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import GuidedBackprop, LayerGradCam, GuidedGradCam, Occlusion, visualization

src_path = os.path.abspath("../src")
if src_path not in sys.path:
    sys.path.append(src_path)
from models import get_model
from train_utils import load_model


def create_combined_visualization(models, image_path, output_dir):
    """Create a grid visualization with all models and methods"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    image_name = os.path.basename(image_path)

    # Order methods as in the example image
    method_names = ['Occlusion', 'Guided Backprop', 'GradCAM', 'Guided GradCAM']

    # Create figure with specific layout
    fig, axes = plt.subplots(len(models), len(method_names), figsize=(16, 5 * len(models)))

    # Set column titles for all columns
    for col, title in enumerate(method_names):
        axes[0, col].set_title(title, fontsize=14)

    # Process each model
    for row, model_name in enumerate(models):
        print(f"Processing {model_name}...")

        # Load model
        model = get_model(model_name, num_classes=7, freeze_layers=False)
        checkpoint_path = f"../models/{model_name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()

        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = outputs.argmax(dim=1).item()
            class_probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence = class_probs[0, pred_class].item() * 100

        # Get class name
        class_names = {
            0: "akiec",
            1: "bcc",
            2: "bkl",
            3: "df",
            4: "mel",
            5: "nv",
            6: "vasc"
        }
        class_name = class_names.get(pred_class, f"Class {pred_class}")

        # Set row label with model name and prediction info
        row_label = f"{model_name}\n({class_name} {confidence:.1f}%)"
        axes[row, 0].set_ylabel(row_label, size='large', rotation=90, labelpad=15)

        # Get last convolutional layer for GradCAM methods
        last_conv = None
        if model_name != "vit_b_16":
            if hasattr(model, 'features'):  # For vgg, densenet, efficientnet
                last_conv = model.features[-1]
            elif hasattr(model, 'layer4'):  # For resnet
                last_conv = model.layer4[-1]

        # Process each attribution method
        for col, method_name in enumerate(method_names):
            ax = axes[row, col]

            # Remove axis ticks and labels for all plots
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colorbar ticks
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.xaxis.set_tick_params(labelsize=8)

            # Skip GradCAM methods for ViT
            if model_name == "vit_b_16" and (method_name == 'GradCAM' or method_name == 'Guided GradCAM'):
                ax.text(0.5, 0.5, f"Not applicable\nfor vit_b_16",
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                continue

            try:
                # Create appropriate attribution method
                if method_name == 'Occlusion':
                    method = Occlusion(model)
                    attributions = method.attribute(
                        input_tensor,
                        target=pred_class,
                        strides=(3, 8, 8),
                        sliding_window_shapes=(3, 15, 15),
                        baselines=0
                    )
                elif method_name == 'Guided Backprop':
                    method = GuidedBackprop(model)
                    attributions = method.attribute(input_tensor, target=pred_class)
                elif method_name == 'GradCAM':
                    method = LayerGradCam(model, last_conv)
                    attributions = method.attribute(input_tensor, target=pred_class)
                    # Handle upsampling if needed
                    if attributions.shape[2:] != input_tensor.shape[2:]:
                        from torch.nn.functional import interpolate
                        attributions = interpolate(attributions, size=input_tensor.shape[2:],
                                                   mode='bilinear', align_corners=False)
                elif method_name == 'Guided GradCAM':
                    method = GuidedGradCam(model, last_conv)
                    attributions = method.attribute(input_tensor, target=pred_class)

                # Convert tensor to numpy for visualization
                attr_np = attributions.squeeze().cpu().detach().numpy()

                # Handle different attribution shapes
                if len(attr_np.shape) == 3 and attr_np.shape[0] == 3:  # RGB channels first
                    attr_np = np.transpose(attr_np, (1, 2, 0))
                elif len(attr_np.shape) == 2:  # Single channel
                    attr_np = np.expand_dims(attr_np, axis=2)

                # Simple visualization similar to the example image
                if method_name == 'Occlusion':
                    # For Occlusion, we want a direct heatmap with no image overlay
                    if len(attr_np.shape) == 3 and attr_np.shape[2] == 1:
                        attr_viz = attr_np.squeeze()
                    else:
                        attr_viz = np.mean(np.abs(attr_np), axis=2)

                    # Normalize for better visualization
                    attr_min = attr_viz.min()
                    attr_max = attr_viz.max()
                    if attr_max > attr_min:
                        attr_viz = (attr_viz - attr_min) / (attr_max - attr_min)

                    im = ax.imshow(attr_viz, cmap='jet', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)

                elif method_name == 'Guided Backprop' or method_name == 'Guided GradCAM':
                    # For gradient-based methods, take the absolute mean across channels
                    if len(attr_np.shape) == 3 and attr_np.shape[2] == 3:
                        attr_viz = np.mean(np.abs(attr_np), axis=2)
                    else:
                        attr_viz = np.abs(attr_np.squeeze())

                    # Normalize for better visualization
                    attr_min = attr_viz.min()
                    attr_max = attr_viz.max()
                    if attr_max > attr_min:
                        attr_viz = (attr_viz - attr_min) / (attr_max - attr_min)

                    im = ax.imshow(attr_viz, cmap='jet', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)

                elif method_name == 'GradCAM':
                    # For GradCAM, direct heatmap visualization
                    if len(attr_np.shape) == 3 and attr_np.shape[2] == 1:
                        attr_viz = attr_np.squeeze()
                    else:
                        attr_viz = np.mean(np.abs(attr_np), axis=2)

                    # Normalize for better visualization
                    attr_min = attr_viz.min()
                    attr_max = attr_viz.max()
                    if attr_max > attr_min:
                        attr_viz = (attr_viz - attr_min) / (attr_max - attr_min)

                    im = ax.imshow(attr_viz, cmap='jet', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)

            except Exception as e:
                print(f"Error applying {method_name} to {model_name}: {e}")
                ax.text(0.5, 0.5, "Error",
                        ha='center', va='center', transform=ax.transAxes)

    # Adjust layout
    plt.suptitle("XAI Visualization Comparison Across Models", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle

    # Save the figure
    output_path = os.path.join(output_dir, f"xai_comparison_grid_{image_name}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Combined visualization saved to {output_path}")





if __name__ == "__main__":
    models = ["resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b_16"]

    # nv
    #image_path = r"C:\Users\carin\OneDrive - Johns Hopkins\Documents\skin_lesion_classification_explainable_AI\HAM10000\HAM10000_images_part_1\ISIC_0027720.jpg"
    # mel
    image_path = r"C:\Users\carin\OneDrive - Johns Hopkins\Documents\skin_lesion_classification_explainable_AI\HAM10000\HAM10000_images_part_2\ISIC_0033050.jpg"
    output_dir = r"C:\Users\carin\OneDrive - Johns Hopkins\Documents\skin_lesion_classification_explainable_AI\xai_results"

    # Create only the combined grid visualization
    create_combined_visualization(models, image_path, output_dir)