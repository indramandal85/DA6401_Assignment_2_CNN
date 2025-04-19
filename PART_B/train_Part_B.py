# train.py

"""
This script trains a classification model using PyTorch Lightning. 
It includes a custom data module, a configurable model wrapper supporting multiple pretrained CNN backbones, 
and a training routine with early stopping and optional test evaluation.
"""

# ---------------------------------------------
# Imports
# ---------------------------------------------
import os
import gc
import time
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import vit_b_16
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torchinfo import summary



# --------------------------------------------------------------------------------
# Custom Data Module for Dataset Handling
# --------------------------------------------------------------------------------
class CustomDataModule(LightningDataModule):
    """
    This class manages data loading, preprocessing, and augmentation using PyTorch Lightning's DataModule structure.

    Attributes:
        data_dir (str): Path to the dataset root folder.
        image_size (tuple): Dimensions to which all images will be resized.
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of the training data to use for validation.
        use_augmentation (bool): Whether to apply data augmentation during training.
        num_workers (int): Number of subprocesses to use for data loading.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 64,
        val_split: float = 0.2,
        use_augmentation: bool = False,
        num_workers: int = 2,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.use_augmentation = use_augmentation
        self.num_workers = num_workers
        self.seed = seed
        self.class_names = []

        self.train_transform = self._build_train_transform()
        self.test_transform = self._build_test_transform()

    def _build_train_transform(self):
        """
        Builds transformation pipeline for training images.
        If augmentation is enabled, applies random resizing, flipping, and color jittering.
        """
        if self.use_augmentation:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def _build_test_transform(self):
        """
        Builds a standard transformation pipeline for validation/testing images.
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: Optional[str] = None):
        """
        Prepares training and validation datasets by splitting and transforming the input images.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.train_transform)
        self.class_names = full_dataset.classes

        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.val_dataset.dataset.transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self, test_dir: Optional[str] = None):
        """
        Returns a test dataloader, optionally from a separate directory.
        """
        test_path = Path(test_dir) if test_dir else self.data_dir.parent / "val"
        test_dataset = datasets.ImageFolder(root=test_path, transform=self.test_transform)
        return DataLoader(test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


# --------------------------------------------------------------------------------
# LightningModule: Model Definition & Training Logic
# --------------------------------------------------------------------------------
class LitClassifier(LightningModule):
    """
    LightningModule that wraps a pretrained model for classification tasks.
    
    Supports flexible fine-tuning strategies and compatible with multiple popular architectures.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 1e-3,
        finetune_strategy: Literal['freeze_all', 'freeze_partial', 'unfreeze_all'] = 'freeze_all',
        k_layers: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Load selected architecture and replace classifier head
        self.model = self._load_model(model_name, num_classes)

        # Apply fine-tuning strategy
        self._apply_finetune_strategy(finetune_strategy, k_layers)

    def _load_model(self, model_name: str, num_classes: int) -> nn.Module:
        """
        Loads the requested pretrained model and adapts the final classification layer.
        """
        if model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif model_name == 'inception_v3':
            model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.aux_logits = False
        elif model_name == 'googlenet':
            model = models.googlenet(weights='IMAGENET1K_V1', aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.aux_logits = False
        elif model_name == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'vit_b_16':
            model = vit_b_16(weights='IMAGENET1K_V1')
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model

    def _apply_finetune_strategy(self, strategy: str, k_layers: int):
        """
        Applies fine-tuning strategy to freeze/unfreeze model layers as needed.
        """
        layers = list(self.model.children())

        if strategy == 'freeze_all':
            for param in self.model.parameters():
                param.requires_grad = False
            self._unfreeze_final_classifier()
        elif strategy == 'freeze_partial':
            cutoff = len(layers) - k_layers
            for idx, layer in enumerate(layers):
                if idx < cutoff:
                    for param in layer.parameters():
                        param.requires_grad = False
            self._unfreeze_final_classifier()
        elif strategy == 'unfreeze_all':
            for param in self.model.parameters():
                param.requires_grad = True

    def _unfreeze_final_classifier(self):
        """
        Ensures the final classification head is always trainable.
        """
        if hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'heads'):
            for param in self.model.heads.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits[0] if isinstance(logits, tuple) else logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(params, lr=self.lr)


# --------------------------------------------------------------------------------
# Main Training Script
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------
    # Configurable Hyperparameters
    # -------------------------------
    MODEL_NAME = "efficientnet_v2_s"
    USE_AUGMENTATION = True
    FINETUNE_STRATEGY = "freeze_partial"
    K_LAYERS = 1
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    SEED = 42
    NUM_EPOCHS = 1
    DATA_DIR = "/Users/indramandal/Documents/VS_CODE/DA6401/DA6401_Assignment_2/inaturalist_12K/train"

    IMAGE_SIZE = {
        "resnet50": (224, 224),
        "vgg16": (224, 224),
        "inception_v3": (299, 299),
        "googlenet": (224, 224),
        "efficientnet_v2_s": (384, 384),
        "vit_b_16": (224, 224),
    }[MODEL_NAME]

    # -------------------------------
    # System Setup
    # -------------------------------
    seed_everything(SEED)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------
    # Prepare Data
    # -------------------------------
    print("=" * 80)
    print(" Loading Dataset ".center(80))
    print("=" * 80)
    time.sleep(2)

    data_module = CustomDataModule(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        use_augmentation=USE_AUGMENTATION,
        seed=SEED
    )
    data_module.setup(stage="fit")

    print(f"Found {len(data_module.class_names)} classes.")
    print(f"Using image size: {IMAGE_SIZE}")
    time.sleep(1)

    # -------------------------------
    # Initialize Model
    # -------------------------------
    print("\n" + "=" * 80)
    print(" Building Model ".center(80))
    print("=" * 80)
    time.sleep(2)

    model = LitClassifier(
        model_name=MODEL_NAME,
        num_classes=len(data_module.class_names),
        lr=LEARNING_RATE,
        finetune_strategy=FINETUNE_STRATEGY,
        k_layers=K_LAYERS
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("\nModel Architecture Summary:")
    summary(model, input_size=(1, 3, *IMAGE_SIZE))
    time.sleep(3)

    # -------------------------------
    # Train Model
    # -------------------------------
    print("\n" + "=" * 80)
    print(" Training Started ".center(80))
    print("=" * 80)
    time.sleep(2)

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        precision="16-mixed",
        accumulate_grad_batches=2,
        callbacks=[EarlyStopping(monitor="val_acc", patience=3, mode="max")],
    )

    trainer.fit(model, datamodule=data_module)
    print("\nTraining Completed Successfully!")

    # -------------------------------
    # Evaluate on Test Data
    # -------------------------------
    print("\n" + "=" * 80)
    print(" Evaluating on Test Set ".center(80))
    print("=" * 80)
    time.sleep(1)

    test_loader = data_module.test_dataloader()
    test_results = trainer.test(model, dataloaders=test_loader)

    test_acc = test_results[0].get('val_acc')
    if test_acc is not None:
        print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    else:
        print("\nTest accuracy metric not found. Please check if val_acc was logged properly.")
