
# CNN-Based Image Classifier for iNaturalist Dataset
This project focuses on building and experimenting with Convolutional Neural Network (CNN) models for image classification using a subset of the iNaturalist dataset. The project is divided into two parts: training a CNN from scratch and fine-tuning a pre-trained model. This README provides a comprehensive guide to implementing the solution, understanding the code, and reproducing the results.

## Problem Statement
The goal of this project is two fold:

**Part A**: Train a CNN model from scratch, tune hyperparameters, and visualize filters.

**Part B**: Fine-tune a pre-trained model for real-world applications.

This README file focuses on Part A, where we build and experiment with a custom CNN-based image classifier trained on a subset of the iNaturalist dataset.
## **Part A**: Train a CNN model from scratch, tune hyperparameters, and visualize filters.
### Setup Instructions
Prerequisites
Ensure you have the following installed:

* Python 3.8+

* PyTorch 2.0+

* PyTorch Lightning

* torchvision

* wandb

Other required libraries: numpy, matplotlib, Pillow

```http
import torch
import wandb
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
from typing import Optional, Tuple
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchsummary import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
```

### Installation
1. Clone this repository:
```http
  git clone https://github.com/indramandal85/DA6401_Assignment_2_CNN.git

```
2. Install dependencies:
```http
  pip install -r requirements.txt
```
3. Log in to Weights & Biases (wandb):
```http
  import wandb
wandb.login(key='YOUR_API_KEY')
```
4. Download the iNaturalist dataset and place it in the appropriate directory:
```http
  https://storage.googleapis.com/wandb_datasets/nature_12K.zip
```

### **Q1. Train a CNN model from scratch**:
### Implementation Details
#### 1. Data Module (`CustomDataModule`)
The CustomDataModule class handles data loading, transformation, and splitting into training, validation, and test sets.

#### Key Features:
- Flexible image resizing and normalization using torchvision.transforms.
- Optional data augmentation (e.g., random cropping, flipping, color jitter).
- Configurable batch size, number of workers, and random seed.

#### Code Reference:
```http
  data_module = CustomDataModule(
    data_dir="/path/to/dataset/train",
    use_augmentation=True,
    val_split=0.25,
    batch_size=64,
    seed=42,
    image_size=(224, 224)
    )
data_module.setup(stage="fit")
```
### 3. Training
We use PyTorch Lightning's Trainer class to train the model with mixed precision for faster computation.

#### Code Reference:
```http
trainer = Trainer(
    max_epochs=20,
    precision="16-mixed",
    callbacks=[EarlyStopping(monitor="val_acc", patience=5)],
    enable_progress_bar=True
    )
trainer.fit(model, datamodule=data_module)
```
#### 4. Testing
```http
trainer.test(model, datamodule=data_module)
```
### Usage
####  Visualize Results on wandb.ai
Log in to your wandb account to view training metrics and visualizations such as loss curves and accuracy plots.

### Results
#### Model Summary:
Below is an example summary of the custom CNN model:
| Layer Type | Output Shape    | Parameters                |
| :--------  | :-------        | :-------------------------|
| Conv2D + ReLU    | 	[-1, 64, 224, 224] | 1,792 |
| MaxPooling     | [-1, 64, 112, 112] | O |
| Fully Connected    | 	[-1, 256] | 	803,072 |
| Output Layer    | [-1, 10] | 	2,570 |

### **Q2. Hyperparameter Tuning with Weights & Biases (wandb)**

### Dataset
We used the iNaturalist dataset with the following setup:

Training set: 80% of the provided training data

Validation set: 20% of the provided training data (stratified by class)

Test set: Separate untouched test folder
### Hyperparameter Tuning Strategy
We used Bayesian optimization through wandb sweeps to efficiently search the hyperparameter space:
```http
sweep_config = {
    "method": "bayes",  # Bayesian Optimization for efficiency
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "first_layer_filters": {"values": [32, 64, 128]},
        "filter_org": {"values": [1.0, 2.0, 0.5]},
        "conv_layers": {"values": [3, 4, 5]},
        "activation": {"values": ["relu", "gelu", "silu", "mish"]},
        "dropout": {"values": [0.2, 0.3]},
        "batch_norm": {"values": [True, False]},
        "batch_size": {"values": [32, 64]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
        "use_augmentation": {"values": [True, False]},
        "num_neurons_dense": {"values": [128, 256, 512]},
    },
}
sweep_id = wandb.sweep(sweep_config, project="CNN_Hyperparameter_Tuning_3")
```
###  Training Function for Sweep Runs:
```http
def train():
    wandb.init()
    
    # Set descriptive run name based on hyperparameters
    run_name = (f"-ac-{wandb.config.activation}"
                f"-filters-{wandb.config.first_layer_filters}"
                f"-filt_org-{wandb.config.filter_org}"
                f"-conv_layers-{wandb.config.conv_layers}"
                f"-dropout-{wandb.config.dropout}"
                f"-batch_norm-{wandb.config.batch_norm}"
                f"-data_aug-{wandb.config.use_augmentation}"
                f"-num_neurons_dense-{wandb.config.num_neurons_dense}")
    
    wandb.run.name = run_name
    
    # Setup data and model
    data_module = CustomDataModule(...)
    model = CustomCNN(...)
    
    # Configure training with early stopping
    early_stop = EarlyStopping(monitor="val_acc", patience=5, mode="max")
    
    trainer = Trainer(
        max_epochs=50,
        precision="16-mixed",  
        callbacks=[early_stop],
        enable_progress_bar=True,
        logger=wandb_logger
    )
    
    trainer.fit(model, datamodule=data_module)

```

### Hyperparameter Performance (wandb Results)

#### Accuracy vs. Created Plot:
- Multiple experiments yielded accuracies between 0.1 and 0.4

- Best configurations achieved ~0.43 ( 43% ) validation accuracy

- Several experimental runs were conducted

#### Parallel Coordinates Plot

#### Training and Validation Metrics

- Val_loss

- Val_acc 

- Train_loss 

- Train_acc

### Best Hyperparameter Configuration
Based on the sweep results, the optimal configuration is:

- First layer filters: 128

- Filter organization: 1.0

- Convolutional layers: 5

- Activation function: RELU

- Dropout rate: 0.3

- Batch normalization: True

- Data augmentation: True

- Learning rate: 1e-3

### Below the wandb report link is added:
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)


 ### **Q3. CNN Hyperparameter Tuning Analysis: iNaturalist Dataset**

###  Key Takeaways :
####  More filters in initial layers (64, 128) lead to better feature extraction and higher validation accuracy.
This parameter choice is impactful
```http
"first_layer_filters": {"values": [32, 64, 128]}

```
####  Batch normalization significantly improves generalization, stabilizing both training and validation loss.
```http
"batch_norm": {"values": [True, False]}
```
####   ReLU and SiLU activations work better than Mish and GELU in this dataset.
RELU emerges as optimal choice from these options
```http
"activation": {"values": ["relu", "gelu", "silu", "mish"]}
```
####  Gradually increasing the number of filters (doubling strategy) is better than halving.
More layers capture hierarchical features better
Models with 5 convolutional layers consistently outperform shallower networks
```http
"conv_layers": {"values": [3, 4, 5]}
```
####  A dropout rate of 0.3 helps prevent overfitting, whereas 0.2 is not as effective.
```http
"dropout": {"values": [0.2, 0.3]},

```
####  Using too few filters (32) and reducing them in deeper layers (filter_org=0.5) leads to poor performance.
Uniform filter distribution works well
```http
"filter_org": {"values": [1.0, 2.0, 0.5]}
```
####  Models with fluctuating validation loss and increasing loss after many epochs indicate overfitting or poor generalization.
```http
"first_layer_filters": {"values": [32, 64, 128]}

```
####  Models trained with augmentation show better validation performance despite potentially higher training loss
Enabling augmentation improves generalization
```http
"use_augmentation": {"values": [True, False]}
```
####  Larger dense layers (512 neurons) generally perform better than smaller ones.
Larger dense layers capture class distinctions better
```http
"num_neurons_dense": {"values": [128, 256, 512]}
```

### Summary
The hyperparameter sweep reveals that optimal CNN performance on the iNaturalist dataset requires balancing model capacity (through filter counts and network depth) with effective regularization (batch normalization and dropout). RELU activation provides performance benefits over other functions, and data augmentation is essential for generalization on this natural image dataset.

These insights demonstrate the importance of thorough hyperparameter tuning for achieving optimal CNN performance, with the best configuration achieving approximately 0.43 validation accuracy on this challenging fine-grained classification task.

### Below the report link of wandb report is added:
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)


### **Q4. CNN-Based Image Classifier on iNaturalist Dataset**

###  Dataset
The project uses a subset of the iNaturalist dataset, containing approximately 12,000 images across 10 biological classification categories.

### Data Augmentation
Augmentation techniques applied to improve model generalization:
```http
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

```
### Optimal Hyperparameters
After extensive experimentation, previously given hyperparameters yielded the best performance:

### Training Process
The model was trained using PyTorch Lightning to streamline the training pipeline:

```http
# Define a checkpoint callback to save best model
checkpoint_callback = ModelCheckpoint(
    dirpath=MODEL_SAVE_DIR,
    filename="best_model",
    save_top_k=1,
    monitor="val_acc",
    mode="max"
)

# Train Model
trainer = pl.Trainer(
    max_epochs=optimal_hyperparams["num_epochs"],
    precision=16,  # Mixed-precision for speed
    callbacks=[checkpoint_callback],
    enable_progress_bar=True
)

trainer.fit(model, datamodule=data_module)
```
### Key training features:

- Mixed-precision training (16-bit) for faster computation

- Early stopping to prevent overfitting

- Model checkpointing to save best performing model

- Efficient data loading with DataLoaders

###  Evaluation Results
The model was evaluated on a separate test dataset to assess its generalization capability:
```http
# Evaluate the model on test set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
```
Using the saved best model hyperparameters and this function, i got the test accuracy of 0.44

#### Key Performance Insights
Based on the visualization results:

- Good performance on Plantae class (shown in green with correct predictions)

- Some confusion between visually similar classes (e.g., Animalia misclassified as Reptilia)

- Confidence scores generally higher for correct predictions

- Challenging examples show the model's limitations in distinguishing fine-grained biological features

### Visualization
The project includes a sophisticated visualization of test predictions in a grid format:
```http
# Create a 10x3 grid of images with predictions
fig, axs = plt.subplots(10, 3, figsize=(15, 50))

# Highlight incorrect predictions with a red box
if pred_idx != class_idx:
    rect = patches.Rectangle(
        (0, 0), img_array.shape[1], img_array.shape[0], linewidth=5, edgecolor="red", facecolor="none"
    )
    axs[row_idx, col_idx].add_patch(rect)

# Display both true and predicted labels with confidence
title_color = "lime" if pred_idx == class_idx else "red"
background_color = "green" if pred_idx == class_idx else "darkred"
title_text = f"True: {class_names[class_idx]}\nPred: {pred_label}\n({confidence:.1f}%)"
```

### Integration with Weights & Biases
The project leverages Weights & Biases for experiment tracking and visualization. Below the report link of wandb report is added:
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)


## 💡 Future Improvements ( **Part - B** )

- Fine-tune a pre-trained model (ResNet, EfficientNet) on iNaturalist.
