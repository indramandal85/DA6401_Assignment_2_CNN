# train.py

# This script trains a custom CNN using PyTorch Lightning for image classification.
# It defines:
# 1. A modular data loader class for handling training, validation, and test data.
# 2. A customizable CNN model.
# 3. The training and evaluation logic using PyTorch Lightning's Trainer API.

# ------------------------------------------------------------------------------
# Imports: Required libraries for data handling, model building, and training
# ------------------------------------------------------------------------------

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image



# ------------------------------------------------------------------------------
# CustomDataModule: Handles data loading, splitting, and transformation
# ------------------------------------------------------------------------------

# This class loads images from a given directory, applies preprocessing and
# augmentations, and provides PyTorch-compatible DataLoaders for training,
# validation, and testing. Uses PyTorch Lightning’s standardized DataModule structure.

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_directory: str,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 64,
        valid_split: float = 0.2,
        use_augment: bool = False,
        workers_numbers: int = 2,
        seed: int = 42
    ):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.img_size = img_size
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.use_augment = use_augment
        self.workers_numbers = workers_numbers
        self.seed = seed
        self.class_names = []

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    # Define training transformations including augmentations if enabled
    def _get_train_transform(self):
        if self.use_augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.img_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # Define consistent transforms for validation and test sets
    def _get_test_transform(self):
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load and split dataset into training and validation sets
    def setup(self, stage: Optional[str] = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        full_dataset = datasets.ImageFolder(root=self.data_directory, transform=self.train_transform)
        self.class_names = full_dataset.classes

        total_size = len(full_dataset)
        val_size = int(total_size * self.valid_split)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.val_dataset.dataset.transform = self.test_transform  # Apply test transforms to val set

    # DataLoaders for training, validation, and testing
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.workers_numbers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.workers_numbers, persistent_workers=True)

    def test_dataloader(self, test_dir: Optional[str] = None):
        test_path = Path(test_dir) if test_dir else self.data_directory.parent / "val"
        test_dataset = datasets.ImageFolder(root=test_path, transform=self.test_transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.workers_numbers)



# ------------------------------------------------------------------------------
# CustomCNN: Modular convolutional neural network using PyTorch Lightning
# ------------------------------------------------------------------------------

# This class defines a configurable CNN model with optional batch normalization,
# activation function choices, dropout, and optimizer options. It supports
# training, validation, and testing with PyTorch Lightning.

class CustomCNN(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
        first_layer_filters: int = 32,
        filter_org: float = 2.0,
        kernel_size: int = 3,
        conv_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.3,
        batch_norm: bool = True,
        dense_size: int = 128,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.activation_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish
        }[activation]

        # Build convolutional layers dynamically
        layers = []
        in_channels = input_shape[0]
        filters = first_layer_filters

        for _ in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, filters, kernel_size, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(self.activation_fn())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = filters
            filters = int(filters * filter_org)

        self.conv_block = nn.Sequential(*layers)

        # Determine size of output feature map
        feature_map_size = self._get_conv_output_size(input_shape)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * feature_map_size * feature_map_size, dense_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_size, num_classes)

    def _get_conv_output_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv_block(x)
        return x.shape[2]

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation_fn()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # Shared logic for train/val/test steps
    def _step(self, batch, step_name: str):
        images, labels = batch
        labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(f"{step_name}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizers = {"adam": optim.Adam, "sgd": optim.SGD}
        if self.hparams.optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        optimizer = optimizers[self.hparams.optimizer_name](self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]



# ------------------------------------------------------------------------------
# Training Script: Loads data, trains the model, and evaluates on test set
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pl.seed_everything(42)

    # Define dataset path (adjust for your system)
    data_dir = "/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_2/inaturalist_12K/train"

    # ------------------- Initialize Data Module -------------------
    print("=" * 80)
    print(" Initializing Data Module ".center(80))
    print("=" * 80)

    data_module = CustomDataModule(
        data_directory=data_dir,
        img_size=(224, 224),
        batch_size=64,
        valid_split=0.2,
        use_augment=True,
        seed=42
    )
    data_module.setup(stage="fit")

    print("\n Data Module Setup Completed")
    print(f" Number of Classes: {len(data_module.class_names)}")
    print(f" Batch Size: {data_module.batch_size}")
    print(f" Validation Split: {data_module.valid_split * 100:.0f}%\n")

    time.sleep(2)

    # ------------------- Initialize Model -------------------
    print("=" * 80)
    print(" Building Custom CNN Model ".center(80))
    print("=" * 80)

    model = CustomCNN(
        input_shape=(3, 224, 224),
        num_classes=len(data_module.class_names),
        first_layer_filters=64,
        filter_org=1.0,
        kernel_size=3,
        conv_layers=5,
        activation="gelu",
        dropout=0.3,
        batch_norm=True,
        dense_size=512,
        learning_rate=1e-3,
        optimizer_name="adam"
    )

    time.sleep(2)

    # ------------------- Model Summary -------------------
    print("\n Model Summary:")
    print("-" * 80)
    summary(model, input_size=(1, 3, 224, 224))
    print("-" * 80)

    time.sleep(2)

    # ------------------- Training -------------------
    early_stopping = EarlyStopping(monitor="val_acc", patience=5, mode="max")

    print("=" * 80)
    print("  Starting Training ".center(80))
    print("=" * 80)

    trainer = Trainer(
        max_epochs=1,
        precision="16-mixed",
        callbacks=[early_stopping],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)

    print("\n Training Complete")
    print("=" * 80)

    time.sleep(2)

    # ------------------- Test Evaluation -------------------
    print("\n" + "=" * 80)
    print("  Starting Test Evaluation ".center(80))
    print("=" * 80)

    test_loader = data_module.test_dataloader()
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    print("\n Test Set Evaluation Complete")
    print("=" * 80)
    print(f" Test Accuracy: {test_results[0]['test_acc'] * 100:.2f}%")
    print("=" * 80)



    # Initialize wandb
    # wandb.init(project="assignment-2", name="test-results")

    # Get test dataset
    test_loader = data_module.test_dataloader()
    test_dataset = test_loader.dataset  # Access the dataset behind the DataLoader
    class_names = test_dataset.classes

    # Transformation for test images (reuse from data_module)
    test_transform = data_module.test_transform

    # Select 10 random class indices
    np.random.seed(42)
    selected_classes = np.random.choice(len(class_names), 10, replace=False)

    # Set a dark background for better visibility
    plt.style.use("dark_background")

    # -----------------------------------------------
    # Display Title and Selected Class Names
    # -----------------------------------------------

    fig_class, ax_class = plt.subplots(figsize=(12, 3))
    ax_class.set_axis_off()

    selected_class_names = [class_names[idx] for idx in selected_classes]
    class_names_text = " | ".join(selected_class_names)

    ax_class.text(0.5, 0.7, "Test Predictions: Best Model", fontsize=20, fontweight="bold", 
                color="cyan", ha="center", va="center")
    ax_class.text(0.5, 0.3, f"Classes: {class_names_text}", fontsize=12, fontweight="bold", 
                color="white", ha="center", va="center", wrap=True)

    plt.show()

    # -----------------------------------------------
    # Create 10x3 Grid for Visualization
    # -----------------------------------------------

    fig, axs = plt.subplots(10, 3, figsize=(15, 50))
    fig.subplots_adjust(hspace=2.5, wspace=2.5)

    # Iterate over selected classes
    for row_idx, class_idx in enumerate(selected_classes):
        # Find all image paths for this class
        class_samples = [path for path, label in test_dataset.samples if label == class_idx]
        sampled_images = np.random.choice(class_samples, 3, replace=False)

        for col_idx, img_path in enumerate(sampled_images):
            # Load image
            img = Image.open(img_path).convert("RGB")

            # Transform image
            img_tensor = test_transform(img).unsqueeze(0)

            # Predict
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
            pred_idx = torch.argmax(output).item()
            pred_label = class_names[pred_idx]
            confidence = torch.softmax(output, dim=1)[0][pred_idx].item() * 100

            # Convert image for plotting
            img_array = np.array(img)

            # Plot
            axs[row_idx, col_idx].imshow(img_array)
            
            # Highlight incorrect predictions
            if pred_idx != class_idx:
                rect = patches.Rectangle(
                    (0, 0), img_array.shape[1], img_array.shape[0],
                    linewidth=5, edgecolor="red", facecolor="none"
                )
                axs[row_idx, col_idx].add_patch(rect)

            # Title with true/predicted class and confidence
            title_color = "lime" if pred_idx == class_idx else "red"
            background_color = "green" if pred_idx == class_idx else "darkred"
            title_text = f"True: {class_names[class_idx]}\nPred: {pred_label}\n({confidence:.1f}%)"

            axs[row_idx, col_idx].text(
                5, 10, title_text,
                fontsize=12, fontweight="bold", color="white",
                bbox=dict(facecolor=background_color, alpha=0.7, edgecolor='white', boxstyle="round,pad=0.5")
            )

            axs[row_idx, col_idx].axis("off")

    # Final adjustments
    plt.tight_layout()
    # wandb.log({"Test Predictions Grid": wandb.Image(fig)})
    plt.show()
    plt.close(fig)

    # Finish wandb
    # wandb.finish()

