# Comparative Analysis of Neural Network Architectures for Skin Lesion Classification With Insights from Explainable AI

## Project Overview

This project aims to develop a deep learning model for skin lesion classification that is explainable. The model will be trained on the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
 and will be able to provide explanations for its predictions. The goal is to create a model that can accurately classify skin lesions while also providing insights using Captum.
___
## Setup
### Clone Repository
```bash
git clone https://github.com/carinacarino/explainable_skinlesion_classification.git
cd explainable_skinlesion_classification
```
### Requirements
Set up environment with:
```bash
conda create -n skinlesion python=3.9
conda activate skinlesion
```
Then install dependencies with:
```bash
pip install -r requirements.txt
```

### Setup Dataset
Download the HAM10000 dataset from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and extract the contents into the `HAM10000` directory.
Alternatively you can download the dataset from Kaggle [here](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

### Configure the settings
Edit the `config.py` file to set the desired parameters for the model training and evaluation.

### Usage
```bash
python src/main.py
```
___
## Project Structure

```bash
|   .gitignore
|   README.md
|   requirements.txt
|   training.log
|           
+---HAM10000 # Dataset directory
|   |   HAM10000_metadata.csv
|   |   hmnist_28_28_L.csv
|   |   hmnist_28_28_RGB.csv
|   |   hmnist_8_8_L.csv
|   |   hmnist_8_8_RGB.csv
|   |   
|   +---HAM10000_images_part_1
|   |       
|   \---HAM10000_images_part_2
|           
+---images # Images used in the report
|       class_distribution.png
|       confusion_matrix_densenet.png
|       confusion_matrix_efficientnet.png
|       confusion_matrix_vgg16.png
|       confusion_matrix_vit.png
|       gender_dist.png
|       image_sample.png
|       localization_dist.png
|       metrics.png
|       output.png
|       pixel_intensity_hist.png
|       
+---models # Trained models
|       densenet121.pth
|       densenet121_finetuned.pth
|       efficientnet_b0.pth
|       efficientnet_b0_finetuned.pth
|       resnet18.pth
|       resnet18_finetuned.pth
|       vgg16.pth
|       vgg16_finetuned.pth
|       vit_b_16.pth
|       
+---notebooks
|       eda.ipynb
|       
+---results # Model evaluation results
|       densenet121_confusion_matrix.png
|       efficientnet_b0_confusion_matrix.png
|       resnet18_confusion_matrix.png
|       vgg16_confusion_matrix.png
|       
+---src # Source codes
|   |   config.py
|   |   dataloader.py
|   |   ham10000_dataset.py
|   |   main.py
|   |   models.py
|   |   train_utils.py
|   |   xai_vis.py
|   |   __init__.py
|           
\---xai_results # XAI results from Captum
        xai_ISIC_0024308.png
        xai_ISIC_0025624.png
```

## Reference:
https://medium.com/@nahmed3536/exploring-the-ham10000-dataset-355a9c79116b
https://pytorch.org/vision/main/models.html
https://medium.com/pytorch/introduction-to-captum-a-model-interpretability-library-for-pytorch-d236592d8afa
https://github.com/pytorch/captum
https://pytorch.org/tutorials/beginner/introyt/captumyt.html
https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html