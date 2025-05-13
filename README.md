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
+---images # Images used in the report      
+---models # Trained models
+---notebooks    
+---results # Model evaluation results     
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

```

## Reference:
https://medium.com/@nahmed3536/exploring-the-ham10000-dataset-355a9c79116b
https://pytorch.org/vision/main/models.html
https://medium.com/pytorch/introduction-to-captum-a-model-interpretability-library-for-pytorch-d236592d8afa
https://github.com/pytorch/captum
https://pytorch.org/tutorials/beginner/introyt/captumyt.html
https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html