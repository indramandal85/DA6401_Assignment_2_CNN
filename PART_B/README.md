
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