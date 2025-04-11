# Enhanced Vision Transformer with EfficientNet and Masked Autoencoder (MAE)

## Overview
This project implements an enhanced Vision Transformer (ViT) for image classification by pretraining a **Masked Autoencoder (MAE)** using **EfficientNet-B0** as a feature extractor. The pretrained MAE is then fine-tuned for classification tasks. The implementation includes:
- **Self-Supervised Learning** using a Masked Autoencoder (MAE)
- **Transformer-based feature extraction** with Multi-Head Attention
- **EfficientNet-B0 for embedding generation**
- **Training pipeline with PyTorch**
- **Visualization of dataset distributions and augmented samples**
- **Evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)**

## Dependencies
Ensure you have the following installed:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

## Directory Structure
```
project_root/
│── data/
│   ├── train/
│   ├── val/
│   ├── test/
│── model.py
│── train.py
│── test.py
│── README.md
```

## Model Architecture
### Masked Autoencoder (MAE)
- Uses **EfficientNet-B0** for feature extraction
- Randomly masks **75%** of the input features
- Encodes the visible features through **12-layer Transformer blocks**
- Outputs a reconstructed version of the original features

### Enhanced Vision Transformer (ViT)
- Uses the pretrained MAE as the feature extractor
- Outputs **class predictions** using a linear classifier

## Training Pipeline
### 1. Data Preprocessing
- Images are resized to **224x224**
- Normalized to **[-1, 1]** range
- Loaded using PyTorch `DataLoader`

### 2. MAE Pretraining (Self-Supervised Learning)
- Trained to **reconstruct masked features** using MSE loss
- **AdamW optimizer** with a learning rate of `3e-4`
- **10 epochs** of pretraining

### 3. Fine-Tuning for Classification
- The pretrained MAE is used as the feature extractor
- Fully connected layer for classification
- **Cross-Entropy Loss** for training
- **AdamW optimizer** with learning rate `3e-4`
- **10 epochs** of fine-tuning

### 4. Evaluation
- **Accuracy, Precision, Recall, and F1-Score**
- **Confusion Matrix Visualization**

## Running the Project
### Pretraining the MAE
```bash
python train.py --pretrain
```

### Training the Classifier
```bash
python train.py --finetune
```

### Evaluating the Model
```bash
python test.py
```

## Results
After training, the model achieves competitive performance in binary classification tasks. The **confusion matrix** and metrics are displayed after evaluation.

## Visualization
### Class Distribution
The dataset distribution is visualized using Seaborn bar plots.

### Augmented Images
Sample images from the dataset with transformations applied.

### Loss Curves
A plot comparing **training vs validation loss** over epochs.

### Confusion Matrix
A heatmap visualization of the classification results.

## Model Weights
Trained model weights are saved as:
```bash
enhanced_vit_efficientnet_final.pth
```

## Future Work
- Experimenting with **larger EfficientNet versions**
- Using **larger Transformer models** for encoding
- Exploring **unsupervised contrastive learning**

## Contributors
[Your Name]

## License
This project is open-source under the **MIT License**.

