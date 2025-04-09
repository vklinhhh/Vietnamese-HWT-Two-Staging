import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import random
from functools import partial

from utils import strip_vietnamese_diacritics


class VietnameseOCRDataset(Dataset):
    """
    Dataset for Vietnamese OCR with support for two-stage training.
    Provides options for simplified (no diacritics) or full Vietnamese text.
    """
    def __init__(
        self, 
        dataset,
        processor,
        stage: int = 1,  # 1: simplified, 2: full Vietnamese
        max_length: int = 64,
        transform: Optional[Callable] = None,
        keep_original_label: bool = True,
        augmentation_prob: float = 0.3  # Probability of applying augmentations
    ):
        self.dataset = dataset
        self.processor = processor
        self.stage = stage
        self.max_length = max_length
        self.transform = transform
        self.keep_original_label = keep_original_label
        self.augmentation_prob = augmentation_prob
    
    def __len__(self):
        return len(self.dataset)
    
    def _apply_augmentations(self, image):
        """Apply data augmentations with probability"""
        if random.random() > self.augmentation_prob:
            return image
            
        # List of possible augmentations
        augmentations = [
            self._random_rotation,
            self._random_brightness_contrast,
            self._random_blur,
            self._add_noise
        ]
        
        # Apply 1-3 random augmentations
        num_augmentations = random.randint(1, 3)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        for aug_fn in selected_augmentations:
            image = aug_fn(image)
            
        return image
    
    def _random_rotation(self, image):
        """Apply random small rotation"""
        angle = random.uniform(-5, 5)
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    
    def _random_brightness_contrast(self, image):
        """Adjust brightness and contrast"""
        from PIL import ImageEnhance
        
        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        # Random contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        return image
    
    def _random_blur(self, image):
        """Apply slight blur to the image"""
        if random.random() < 0.5:  # 50% chance to apply blur
            from PIL import ImageFilter
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
        return image
    
    def _add_noise(self, image):
        """Add slight noise to the image"""
        from PIL import Image
        import numpy as np
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Add Gaussian noise
        noise_factor = random.uniform(0, 10)
        gaussian_noise = np.random.normal(0, noise_factor, img_array.shape)
        
        # Add noise to image and clip values
        noisy_img = np.clip(img_array + gaussian_noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def __getitem__(self, idx):
        # Load and process just one example at a time
        item = self.dataset[idx]
        image = item.get('image')
        
        # Get the label
        original_label = str(item.get('label', ''))
        
        # For Stage 1: Simplify Vietnamese text by removing diacritics
        if self.stage == 1:
            simplified_label = strip_vietnamese_diacritics(original_label)
            label = simplified_label
        else:
            # For Stage 2: Use the full Vietnamese text with diacritics
            label = original_label
        
        # Apply custom transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Apply data augmentation during training
        if self.stage == 2:  # Apply more augmentations in stage 2
            image = self._apply_augmentations(image)
        
        # Convert image if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Process image (returns dict with pixel_values)
        encoding = self.processor(image, return_tensors="pt")
        # Extract and squeeze the tensor to remove batch dimension
        pixel_values = encoding.pixel_values.squeeze()
        
        # Process text label
        label_encoding = self.processor.tokenizer(
            label, 
            padding="max_length", 
            max_length=min(self.max_length, len(label) + 10),
            truncation=True,
            return_tensors="pt"
        )
        
        # Get the tokens and squeeze to remove batch dimension
        labels = label_encoding.input_ids.squeeze()
        
        # Create result dictionary
        result = {
            "pixel_values": pixel_values,
            "labels": labels
        }

        if self.keep_original_label:
            # Also tokenize original label for Stage 2
            if self.stage == 1:
                original_encoding = self.processor.tokenizer(
                    original_label, 
                    padding="max_length", 
                    max_length=min(self.max_length, len(original_label) + 10),
                    truncation=True,
                    return_tensors="pt"
                )
                result["original_labels"] = original_encoding.input_ids.squeeze()
            
            # Keep text version for evaluation
            result["text_label"] = label
            result["original_text_label"] = original_label
        
        return result


# Custom collator function to handle variable-length sequences
def vietnamese_ocr_collate_fn(batch, pad_token_id):
    # Extract all pixel values and labels
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Handle optional fields
    has_original_labels = "original_labels" in batch[0]
    has_text_labels = "text_label" in batch[0]
    
    # Pad pixel values
    pixel_values = torch.stack(pixel_values)
    
    # Pad labels
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    
    result = {
        "pixel_values": pixel_values,
        "labels": labels
    }
    
    # Include original labels if available
    if has_original_labels:
        original_labels = [item["original_labels"] for item in batch]
        original_labels = torch.nn.utils.rnn.pad_sequence(
            original_labels, 
            batch_first=True, 
            padding_value=pad_token_id
        )
        result["original_labels"] = original_labels
    
    # Include text labels if available
    if has_text_labels:
        result["text_labels"] = [item["text_label"] for item in batch]
        result["original_text_labels"] = [item["original_text_label"] for item in batch]
    
    return result


# Helper function to create dataloaders
def create_vietnamese_ocr_dataloaders(
    train_dataset,
    val_dataset,
    processor,
    stage=1,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    max_length=64
):
    """
    Create dataloaders for training and validation.
    
    Args:
        train_dataset: Raw training dataset
        val_dataset: Raw validation dataset
        processor: TrOCR processor
        stage: Training stage (1 or 2)
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        max_length: Maximum sequence length
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = VietnameseOCRDataset(
        train_dataset,
        processor,
        stage=stage,
        max_length=max_length,
        augmentation_prob=0.5 if stage == 2 else 0.3
    )
    
    val_dataset = VietnameseOCRDataset(
        val_dataset,
        processor,
        stage=stage,
        max_length=max_length,
        augmentation_prob=0  # No augmentation for validation
    )
    
    # Create collate function
    collate_fn = partial(
        vietnamese_ocr_collate_fn,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader