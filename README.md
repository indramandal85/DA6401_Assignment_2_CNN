
# CNN-Based Image Classifier for iNaturalist Dataset
This project focuses on building and experimenting with Convolutional Neural Network (CNN) models for image classification using a subset of the iNaturalist dataset. The project is divided into two parts: training a CNN from scratch and fine-tuning a pre-trained model. This README provides a comprehensive guide to implementing the solution, understanding the code, and reproducing the results.

## Problem Statement
The goal of this project is two fold:

**Part A**: Train a CNN model from scratch, tune hyperparameters, and visualize filters.

**Part B**: Fine-tune a pre-trained model for real-world applications.

In Part A, where we build and experiment with a custom CNN-based image classifier trained on a subset of the iNaturalist dataset.
## **✅Part A**: Train a CNN model from scratch, tune hyperparameters, and visualize filters.
### Setup Instructions
Prerequisites
Ensure you have the following installed:

* Python 3.8+

* PyTorch 2.0+

* PyTorch Lightning

* torchvision

* wandb

Other required libraries: numpy, matplotlib, Pillow

**Below all the Dependencies are provided**
```python
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
```python
  git clone https://github.com/indramandal85/DA6401_Assignment_2_CNN.git

```
2. Install dependencies:
```python
  pip install -r requirements.txt
```
3. Log in to Weights & Biases (wandb):
```python
    import wandb
    wandb.login(key='YOUR_API_KEY')
```
4. Download the iNaturalist dataset and place it in the appropriate directory:
 - [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)

### **Q1. Train a CNN model from scratch**:
### Implementation Details
#### 1. Data Module (`CustomDataModule`)
The CustomDataModule class handles data loading, transformation, and splitting into training, validation, and test sets.

#### Key Features:
- Flexible image resizing and normalization using torchvision.transforms.
- Optional data augmentation (e.g., random cropping, flipping, color jitter).
- Configurable batch size, number of workers, and random seed.

#### Code Reference:
```python
  data_module = CustomDataModule(
    data_dir="/path/to/dataset/train",  # path to training dataset
    use_augmentation=True,              # Augmentation of photos
    val_split=0.20,                     # Validation split is 20%
    batch_size=64,                      # Batch size is choosed as 64
    seed=42,                            # can choose mannual seed also
    image_size=(224, 224)               # Standard image size is choosen
    )
data_module.setup(stage="fit")
```
### 3. Training
We use PyTorch Lightning's Trainer class to train the model with mixed precision for faster computation.

#### Code Reference:
```python
trainer = Trainer(
    max_epochs=20,
    precision="16-mixed",
    callbacks=[EarlyStopping(monitor="val_acc", patience=5)],
    enable_progress_bar=True
    )
trainer.fit(model, datamodule=data_module)
```
#### 4. Testing
```python
trainer.test(model, datamodule=data_module)
```
### Usage
####  Visualize Results on wandb.ai
Click on the Wandb Report Link to view training metrics and visualizations such as loss curves and accuracy plots.
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)

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

- Training set: 80% of the provided training data

- Validation set: 20% of the provided training data (stratified by class)

- Test set: Separate untouched test folder
### Hyperparameter Tuning Strategy
We used Bayesian optimization through wandb sweeps to efficiently search the hyperparameter space:
```python
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
```python
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

#### **Accuracy vs. Created Plot and Accuracy vs. Epochs**:
The “Accuracy vs. Created” plot tracks the validation accuracy of each experiment over time. This plot shows the number of experiments conducted, visually representing the temporal progression as different hyperparameter combinations were explore

- Multiple experiments yielded accuracies between 0.1 and 0.44

- Best configurations achieved ~0.4427 ( 44.27% ) validation accuracy

- Several experimental (more than 480+ sweep) runs were conducted
#### **Parallel Coordinates Plot** :
The parallel coordinates plot visually maps each experiment as a line crossing multiple axes, where each axis represents a different hyperparameter or performance metric.

#### Higher validation accuracies (0.3-0.44) are achieved with:
- GELU / MISH / RELU activation function
- Batch normalization enabled (true)
- Batch sizes around 64
- Conv_layers count of  5
- Learning rates around 0.0001-0.0002
- Dense size between 256-512
- First layer filters 128

#### Training and Validation Metrics

- Val_loss

- Val_acc 

- Train_loss 

- Train_acc
#### **Correlation Summary Table** : (The correlation of each hyperparameter with the accuracy)
- The correlation table reveals critical relationships between hyperparameters and model performance.
- See the detailed analysis in Wandb Report

### Best Hyperparameter Configuration
After completing 480+ wandb sweep runs and analyzing the plots and the correlation summary, there are two best hyperparameter configuration identified for my model which is giving 44.27% of Validation Accuracy. Both of the Best Hyperparameter configuration of my CNN model are provided below:

#### **Configuration - 1** :
- Activation Function:                      gelu

- Batch Normalization:                     true

- Batch Size:                                           64

- Number of Convolutional Layers:  5

- Dense layer size:                                512

- Dropout Rate:                                      0.3

- Filter Organization:                             1

- First Layer Filters:                             128

- Input Size:                                           ( 3, 224, 224)

- Kernel Size:                                         3

- Learning Rate:                                   0.0001

- Number of Output Classes:             10

- Dense Layer Neurons:                      512

- Optimizer:                                           adam

- Augmentation:                                   false



#### **Configuration - 2** :
- Activation Function:                       mish

- Batch Normalization:                     true

- Batch Size:                                           64

- Number of Convolutional Layers:  5

- Dense layer size:                                512

- Dropout Rate:                                      0.3

- Filter Organization:                             1

- First Layer Filters:                             128

- Input Size:                                           ( 3, 224, 224)

- Kernel Size:                                         3

- Learning Rate:                                   0.0001

- Number of Output Classes:             10

- Dense Layer Neurons:                      512

- Optimizer:                                           adam

- Augmentation:                                   false




### Below the wandb report link is added:
**Full Report can be explored on wandb**
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)


 ### **Q3. CNN Hyperparameter Tuning Analysis: iNaturalist Dataset**

###  Key Takeaways :
####  More filters in initial layers (64, 128) lead to better feature extraction and higher validation accuracy.
This parameter choice is impactful
```python
"first_layer_filters": {"values": [32, 64, 128]}

```
####  Batch normalization significantly improves generalization, stabilizing both training and validation loss.
```python
"batch_norm": {"values": [True, False]}
```
####   ReLU and SiLU activations work better than Mish and GELU in this dataset.
RELU emerges as optimal choice from these options
```python
"activation": {"values": ["relu", "gelu", "silu", "mish"]}
```
####  Gradually increasing the number of filters (doubling strategy) is better than halving.
More layers capture hierarchical features better
Models with 5 convolutional layers consistently outperform shallower networks
```python
"conv_layers": {"values": [3, 4, 5]}
```
####  A dropout rate of 0.3 helps prevent overfitting, whereas 0.2 is not as effective.
```python
"dropout": {"values": [0.2, 0.3]},

```
####  Using too few filters (32) and reducing them in deeper layers (filter_org=0.5) leads to poor performance.
Uniform filter distribution works well
```python
"filter_org": {"values": [1.0, 2.0, 0.5]}
```
####  Models with fluctuating validation loss and increasing loss after many epochs indicate overfitting or poor generalization.
```python
"first_layer_filters": {"values": [32, 64, 128]}

```
####  Models trained with augmentation show better validation performance despite potentially higher training loss
Enabling augmentation improves generalization
```python
"use_augmentation": {"values": [True, False]}
```
####  Larger dense layers (512 neurons) generally perform better than smaller ones.
Larger dense layers capture class distinctions better
```python
"num_neurons_dense": {"values": [128, 256, 512]}
```

### Summary
The hyperparameter sweep reveals that optimal CNN performance on the iNaturalist dataset requires balancing model capacity (through filter counts and network depth) with effective regularization (batch normalization and dropout). RELU activation provides performance benefits over other functions, and data augmentation is essential for generalization on this natural image dataset.

These insights demonstrate the importance of thorough hyperparameter tuning for achieving optimal CNN performance, with the best configuration achieving approximately 0.43 validation accuracy on this challenging fine-grained classification task.

### Below the wandb report link is added:
**Full Report can be explored on wandb**
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)


### **Q4. CNN-Based Image Classifier on iNaturalist Dataset**

###  Dataset
The project uses a subset of the iNaturalist dataset, containing approximately 12,000 images across 10 biological classification categories.

### Data Augmentation
Augmentation techniques applied to improve model generalization:
```python
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

```python
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
```python
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
Using the saved best model hyperparameters and this function, the test accuracy of CustomCNN model is 0.4427 (**44.27%**)

#### Key Performance Insights:
Based on the visualization results:

- Good performance on Plantae class (shown in green with correct predictions)

- Some confusion between visually similar classes (e.g., Animalia misclassified as Reptilia)

- Confidence scores generally higher for correct predictions

- Challenging examples show the model's limitations in distinguishing fine-grained biological features

### Visualization
The project includes a sophisticated visualization of test predictions in a 10 * 3 grid format:
```python
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

#### Integration with Weights & Biases:
The project leverages Weights & Biases for experiment tracking and visualization. Below the report link of wandb report is added:
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)





### Visulation of all the filters of Convolution Layers of the best model for a random image in an 8×8 grid:
- In this part of the code, proper Visualazation of all 5 conv layers filters in 8*8 grid have been done properly.
- This visuallation helps to give better understanding about conv layers filters and how the CNN collects the spatial features inside any grid based data.
- To Visualize these layers and also proper discussion about every layers and their filters, see the wandb report


### Guided back-propagaation on random10 neurons in the CONV5 layer and ploted the images of excited neurons:
- Here in this part The guided gradient visualizations reveal distinct patterns that each neuron responds to.
- In Wandb Report, the discussion about every neurons and their special features capturing capabilities are discussed in detail.


### Below the wandb report link is added for detailed report:

 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)



## **✅PART - B:** Fine-tuning Pre-trained ImageNet Models for Naturalist Dataset Classification: 

This part of repository implements transfer learning techniques to adapt pre-trained ImageNet models for a naturalist dataset classification task. The project addresses key challenges when working with pre-trained models and demonstrates effective fine-tuning strategies.

### **Q.1: Fine-Tuning Pre-Trained Models for the Naturalist Dataset**

This document outlines the implementation details for adapting pre-trained ImageNet models to the Naturalist dataset (10 classes) by addressing **image dimension mismatches** and **class mismatch** in the output layer.

---

### ➤ Image Dimension Handling :

### Problem Statement  
Pre-trained ImageNet models expect specific input dimensions (e.g., `224x224` for ResNet/VGG or `299x299` for InceptionV3). The Naturalist dataset may have images of varying sizes.

### Solution  
The `CustomDataModule` class dynamically resizes images to match the target model's requirements using PyTorch's `transforms` pipeline:

#### Key Features:
- **Resizing**:  
  - Training images are resized to `224x224` (default) or `299x299` (if using InceptionV3).  
  - Augmentation is applied via `RandomResizedCrop` while maintaining target dimensions.  
- **Normalization**:  
  Uses ImageNet's standard mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` for consistency.  

#### Code Snippet:
```python
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, image_size: Tuple[int, int] = (224, 224), ...):
        self.image_size = image_size  # Set to (299, 299) for InceptionV3
    
    def _get_train_transform(self):
        if self.use_augmentation:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),  # Maintains target size
                ...
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),  # Simple resizing
                ...
            ])
```            

### ➤ Strategey to address the Last Layers Nodes Mismatch :  
Pre-trained models have a final classification layer with 1000 nodes (ImageNet classes), but the Naturalist dataset requires 10 outputs.


### Solution  
The `LitClassifier` class replaces the final layer of the pre-trained model with a new fully connected layer of size 10, preserving feature extraction capabilities.

 **Architecture-Specific Modifications:**
| Model | 	Layer Replacement Code    |
| :--------  | :-------        |
| ResNet50    | 	model.fc = nn.Linear(... , 10) |
| VGG16     | model.classifier[6] = nn.Linear(... , 10) |
| Inception_v3    | 		model.fc = nn.Linear(... , 10) |
| EfficientNetV2    | model.classifier[1] = nn.Linear(... , 10)|



```python
class LitClassifier(pl.LightningModule):
    def _load_model(self, model_name, num_classes):
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.aux_logits = False  # Disable auxiliary logits
        ...
        return model
```

### Implementation Workflow
1. Data Preparation:

- Initialize CustomDataModule with image_size matching the target model (e.g., 224x224 for ResNet, 299x299 for InceptionV3).

- Apply resizing, normalization, and optional augmentation.

2. Model Setup:

- Load a pre-trained model via LitClassifier, which automatically replaces the final layer.

- Choose a fine-tuning strategy (freeze_all, freeze_partial, unfreeze_all).

3. Training:

- Use PyTorch Lightning's Trainer to execute fine-tuning.

### Full Report can be explored on wandb :
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)


### **Q.2: Fine-Tuning Pre-Trained Models Using Layer Freezing Strategies**
**Layer Freezing Strategies for Efficient Transfer Learning**

---

### Overview
This project demonstrates efficient fine-tuning of large pre-trained models (GoogLeNet, InceptionV3, ResNet50, VGG16, EfficientNetV2) on the iNaturalist dataset using **layer freezing strategies**. Large models are computationally expensive to train, and layer freezing enables tractable training while preserving pre-learned features. Three key strategies are implemented:

1. **Freezing All Layers Except the Classifier**  
2. **Freezing Early Layers (Partial Freezing)**  
3. **Full Fine-Tuning (Unfreezing All Layers)**  

---

### Implementation Details :

#### Key Components
- **Models Supported**:  
  GoogLeNet, InceptionV3, ResNet50, VGG16, EfficientNetV2, VisionTransformer
- **Dataset**: iNaturalist (custom directory structure compatible with `ImageFolder`)
- **Core Classes**:  
  - `CustomDataModule`: Handles dataset loading, augmentation, and splitting
  - `LitClassifier`: Implements model loading, layer freezing strategies, and training logic

---

### Layer Freezing Strategies

#### 1. Freeze All Layers Except Classifier (`freeze_all`)
- **Behavior**:  
  - Freezes all layers except the final classification layer
  - Preserves pre-trained feature extractors
  - Ideal for small datasets or tasks with similar low-level features to the pre-training data
- **Code Snippet**:
```python
  def _apply_finetune_strategy(self, strategy, k_layers):
      if strategy == 'freeze_all':
          for param in self.model.parameters():
              param.requires_grad = False
          self._unfreeze_final_classifier()  # Unfreezes classifier

```

#### 2. Freeze Early Layers (`freeze_partial`)
- **Behavior**:  
  - Freezes the first N - k layers (early layers)
  - Unfreezes k layers closest to the output
  - Balances feature reuse and task-specific adaptation

- **Usage**:  Specify `k_layers` (number of unfrozen layers)
- **Code Snippet**:
```python
  elif strategy == 'freeze_partial':
    for i, layer in enumerate(all_layers):
        if i < len(all_layers) - k_layers:  # Freeze early layers
            for param in layer.parameters():
                param.requires_grad = False
    self._unfreeze_final_classifier()
```


#### 3. Full Fine-Tuning (`unfreeze_all`)
- **Behavior**:  
  - Unfreezes all layers for end-to-end training
  - Uses a low learning rate to avoid catastrophic forgetting
  - Requires large datasets and significant computational resources

- **Code Snippet**:
```python
  elif strategy == 'unfreeze_all':
    for param in self.model.parameters():
        param.requires_grad = True
```

#### **Implementation Architecture**
- **Custom Data Pipeline**:  
The CustomDataModule class handles dataset loading and augmentation:
```python
  datamodule = CustomDataModule(
    data_dir='/path/to/inaturalist',
    image_size=(299, 299),  # Model-specific dimensions
    batch_size=32,
    use_augmentation=True,
    val_split=0.15
)
```

- **Model Configuration**:  
The LitClassifier abstracts model loading and freezing logic:
```python
  class LitClassifier(pl.LightningModule):
    def _apply_finetune_strategy(self, strategy, k_layers):
        # Freezing logic implementation
        if strategy == 'freeze_all':
            for param in self.model.parameters():
                param.requires_grad = False
            self._unfreeze_final_classifier()

```

- **Training Execution**: 
```python
  from pytorch_lightning import Trainer

model = LitClassifier(
    model_name='resnet50',
    num_classes=10,
    finetune_strategy='freeze_partial',
    k_layers=2
)

trainer = Trainer(
    max_epochs=20,
    accelerator='gpu',
    devices=1
)

trainer.fit(model, datamodule)
```

- **Conclusion and Recommendations**: 
For resource-constrained environments, it is recommended to start with 2-3 unfrozen layers and progressively increasing model capacity if performance plateaus. The implementation provides a flexible framework for balancing computational requirements with model accuracy through its configurable freezing strategies.


### **Q.3: Transfer Learning Study: Fine-Tuning vs Training from Scratch**
### Overview
This project compares the performance of fine-tuning pre-trained models against training a custom CNN from scratch on a naturalist dataset. The study evaluates three fine-tuning strategies (`freeze_all`, `freeze_partial`, `unfreeze_all`) across multiple architectures (ResNet50, VGG16, InceptionV3, EfficientNet-v2-S, GoogLeNet) and analyzes key metrics such as validation accuracy, convergence speed, and hyperparameter sensitivity.

---

### Key Findings
- **Fine-Tuning Outperforms Scratch Training**:  
  Fine-tuned models achieved **35–82.79% validation accuracy**, while the custom CNN trained from scratch reached only **5–44.27%**.
- **EfficientNet-v2-S Dominates**:  
  Achieved the highest accuracy (up to **85%**) with optimal hyperparameters.
- **Faster Convergence**:  
  Fine-tuned models required **10–20 epochs** to reach peak performance vs slower convergence for scratch training.
- **Hyperparameter Sensitivity**:  
  Learning rates in **0.0003–0.0008** worked best for partial freezing strategies.

---

### Experimental Results

Performance Comparison:
| Metric | Custom CNN (Scratch)    | Fine-Tuned Models (Partial Freeze)  |
| :--------  | :-------        | :-------------------------|
| Validation Accuracy   | 		5–44.27% | **	35–82.79%** |
| Training Epochs    | >30 | 	**	10–20** |
| Hyperparameter Tuning Complexity    | 	High | 	Moderate (LR-sensitive) |

### Key Findings
- This study validates that fine-tuning, especially using partial freezing, offers substantial benefits over training from scratch. It improves generalization, reduces training time, and allows for efficient resource usage, especially when domain-specific data is limited.
---

### **Wandb Report is Suggested to explore for better Understanding, Observations and Experiment details**
**Full Report Link is Provided Below**
 - [Wandb Report](https://wandb.ai/ed24s014-indian-institute-of-technology-madras/CNN_Hyperparameter_Tuning_3/reports/Copy-of-DA6401-Assignment-2--VmlldzoxMjI2MjQyNg?accessToken=n5t71n8616wbg2go0svn1bhi8y7vznqoomthqhpy7wkiqpajhpn7e4ywxx9xodyc)