import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    get_linear_schedule_with_warmup
)
import logging
from tqdm import tqdm
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union, Any

from dataset import create_vietnamese_ocr_dataloaders
from accent_dataset import build_accent_dataloaders, prepare_corpus_dataset
from accent_model import (
    VietnameseAccentClassifier, 
    VietnameseAccentProcessor,
    create_accent_label_map
)
from integrated_model import TwoStageVietnameseOCR
from utils import strip_vietnamese_diacritics, compute_diacritic_metrics


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ocr_stage1(
    model,
    processor,
    train_dataset,
    val_dataset,
    output_dir="./vietnamese-ocr-stage1",
    batch_size=8,
    learning_rate=5e-5,
    num_epochs=5,
    warmup_steps=0.1,
    weight_decay=0.01,
    device=None,
    max_grad_norm=1.0,
    fp16=False
):
    """
    Train OCR model for Stage 1 (without diacritics).
    
    Args:
        model: OCR model
        processor: OCR processor
        train_dataset, val_dataset: Training and validation datasets
        output_dir: Directory to save model
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        warmup_steps: Warm-up steps ratio
        weight_decay: Weight decay for optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use mixed precision training
        
    Returns:
        Dict with training history and best model path
    """
    logger.info("Starting OCR training for Stage 1 (without diacritics)")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_vietnamese_ocr_dataloaders(
        train_dataset,
        val_dataset,
        processor,
        stage=1,  # Without diacritics
        batch_size=batch_size
    )
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    if isinstance(warmup_steps, float):
        warmup_steps = int(total_steps * warmup_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cer": []
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            

            # Instead of filtering to only encoder parameters
            encoder_safe_batch = {k: v for k, v in batch.items() 
                                if k in ['pixel_values', 'attention_mask', 'token_type_ids', 'labels']}
            tokenizer = processor.tokenizer
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.decoder.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.eos_token_id = tokenizer.eos_token_id

            # Forward pass
            optimizer.zero_grad()
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**encoder_safe_batch)
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**encoder_safe_batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Accumulate loss
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                encoder_safe_batch = {k: v for k, v in batch.items() 
                                if k in ['pixel_values', 'attention_mask', 'token_type_ids', 'labels']}
                # Forward pass
                outputs = model(**encoder_safe_batch)
                loss = outputs.loss
                
                # Accumulate loss
                val_loss += loss.item()
                
                # Generate predictions
                generated_ids = model.generate(
                    pixel_values=encoder_safe_batch["pixel_values"],
                    max_length=128,
                    num_beams=5
                )
                
                # Decode predictions and labels
                pred_str = processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                label_str = []
                for label_ids in encoder_safe_batch["labels"]:
                    # Replace -100 with pad token id
                    label_ids = label_ids.clone()
                    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                    label_str.append(processor.tokenizer.decode(
                        label_ids, skip_special_tokens=True
                    ))
                
                # Save predictions and labels
                all_preds.extend(pred_str)
                all_labels.extend(label_str)
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Calculate CER (Character Error Rate)
        import editdistance
        total_distance = sum(editdistance.eval(p, l) for p, l in zip(all_preds, all_labels))
        total_length = sum(len(l) for l in all_labels)
        cer = total_distance / max(1, total_length)
        history["val_cer"].append(cer)
        logger.info(f"Validation CER: {cer:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, f"best_model_epoch{epoch+1}")
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}")
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Return training history and best model path
    return {
        "history": history,
        "best_model_path": best_model_path
    }


def train_accent_stage2(
    corpus_file,
    mapping_file,
    output_dir="./vietnamese-accent-stage2",
    pretrained_model="xlm-roberta-base",
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    warmup_steps=0.1,
    weight_decay=0.01,
    device=None,
    max_grad_norm=1.0,
    fp16=False,
    max_length=128,
    split_ratio=0.9,
    use_crf=False
):
    """
    Train accent model for Stage 2 (accent insertion).
    
    Args:
        corpus_file: Path to corpus file with accented Vietnamese text
        mapping_file: Path to mapping file with accent mappings
        output_dir: Directory to save model
        pretrained_model: Pretrained model name for XLM-RoBERTa
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        warmup_steps: Warm-up steps ratio
        weight_decay: Weight decay for optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use mixed precision training
        max_length: Maximum sequence length
        split_ratio: Train/val split ratio
        use_crf: Whether to use CRF layer
        
    Returns:
        Dict with training history and trained model
    """
    logger.info("Starting accent model training for Stage 2 (accent insertion)")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create accent label mapping
    accent_map, label_to_accented, base_word_map = create_accent_label_map(mapping_file)
    logger.info(f"Created accent mapping with {len(accent_map)} entries")
    # Load corpus data
    corpus_data = prepare_corpus_dataset(corpus_file, split_ratio=split_ratio)
    logger.info(f"Loaded corpus with {len(corpus_data['train'])} training and {len(corpus_data['val'])} validation examples")
    
    # Create accent processor
    processor = VietnameseAccentProcessor(
    tokenizer_name=pretrained_model,
    accent_label_map=accent_map,
    accent_base_word_map=base_word_map,  
    max_length=max_length
    )
    # Create accent model
    model = VietnameseAccentClassifier(
        pretrained_model_name=pretrained_model,
        num_accent_labels=len(accent_map),
        accent_label_map=accent_map,
        dropout_rate=0.1,
        use_crf=use_crf
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create dataloaders
    train_loader = build_accent_dataloaders(
        texts=corpus_data["train"],
        processor=processor,
        batch_size=batch_size,
        max_length=max_length,
        is_training=True,
        use_tagging_dataset=True
    )
    
    val_loader = build_accent_dataloaders(
        texts=corpus_data["val"],
        processor=processor,
        batch_size=batch_size,
        max_length=max_length,
        is_training=True,
        use_tagging_dataset=True
    )
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    if isinstance(warmup_steps, float):
        warmup_steps = int(total_steps * warmup_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            optimizer.zero_grad()
            
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"]
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Accumulate loss
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        logger.info(f"Train loss: {train_loss:.4f}")


    # Validation
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Accumulate loss
            val_loss += loss.item()
            
            # Calculate accuracy
            predictions = model.decode_accents(logits, batch["attention_mask"])
            
            # Convert to flat array for accuracy calculation
            labels = batch["accent_labels"].view(-1)
            
            # Handle different prediction formats based on CRF or standard decoding
            if isinstance(predictions, list):  # CRF case
                # For CRF, predictions is a list of lists of label IDs
                # Convert to a flattened tensor
                flat_preds = []
                for pred_seq in predictions:
                    flat_preds.extend(pred_seq)
                predictions_tensor = torch.tensor(flat_preds, device=device)
            else:
                # For standard decoding, predictions is already a numpy array
                # Convert to tensor and flatten
                predictions_tensor = torch.tensor(predictions, device=device).view(-1)
            
            # Only consider non-ignored positions
            mask = (labels != -100)
            labels_filtered = labels[mask]
            
            # Make sure predictions_tensor has the same shape as labels
            if len(predictions_tensor) < len(labels):
                # Pad predictions if needed
                padding = torch.full((len(labels) - len(predictions_tensor),), -100, 
                                    device=device, dtype=torch.long)
                predictions_tensor = torch.cat([predictions_tensor, padding])
            elif len(predictions_tensor) > len(labels):
                # Truncate predictions if needed
                predictions_tensor = predictions_tensor[:len(labels)]
            
            predictions_filtered = predictions_tensor[mask]
            
            # Calculate accuracy
            correct_preds += (predictions_filtered == labels_filtered).sum().item()
            total_preds += len(labels_filtered)

        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = correct_preds / max(1, total_preds)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        logger.info(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best model found at epoch {epoch+1} with loss {val_loss:.4f}")
            print(output_dir)
            best_model_path = os.path.join(output_dir, f"best_model_epoch{epoch+1}")
            # Save model and processor
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            logger.info(f"Saved best model to {best_model_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}")
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Return training history and best model path
    return {
        "history": history,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "model": model,
        "processor": processor
    }


def train_integrated_model(
    ocr_model_dir,
    accent_model_dir,
    train_dataset,
    val_dataset,
    corpus_file,
    mapping_file,
    output_dir="./vietnamese-ocr-integrated",
    ocr_batch_size=8,
    accent_batch_size=16,
    ocr_lr=5e-5,
    accent_lr=2e-5,
    ocr_epochs=3,
    accent_epochs=3,
    warmup_steps=0.1,
    weight_decay=0.01,
    device=None,
    max_grad_norm=1.0,
    fp16=False,
    max_length=128
):
    """
    Train the complete two-stage Vietnamese OCR model.
    
    Args:
        ocr_model_dir: Directory of pretrained OCR model or name
        accent_model_dir: Directory of pretrained accent model or None
        train_dataset, val_dataset: Training and validation datasets for OCR
        corpus_file: Path to corpus file with accented Vietnamese text
        mapping_file: Path to mapping file with accent mappings
        output_dir: Directory to save the integrated model
        ocr_batch_size: Batch size for OCR training
        accent_batch_size: Batch size for accent training
        ocr_lr: Learning rate for OCR model
        accent_lr: Learning rate for accent model
        ocr_epochs: Number of epochs for OCR training
        accent_epochs: Number of epochs for accent training
        warmup_steps: Warm-up steps ratio
        weight_decay: Weight decay for optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use mixed precision training
        max_length: Maximum sequence length
        
    Returns:
        Dict with training history and final integrated model
    """
    logger.info("Starting training for integrated Vietnamese OCR model")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1: Train OCR model for unaccented Vietnamese
    logger.info("=== Stage 1: Training OCR model for unaccented Vietnamese ===")
    
    # Load or create OCR model
    try:
        # Try to load pretrained model
        ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_dir)
        ocr_processor = TrOCRProcessor.from_pretrained(ocr_model_dir)
        logger.info(f"Loaded pretrained OCR model from {ocr_model_dir}")
    except:
        # Create new model
        logger.info(f"Creating new OCR model from {ocr_model_dir}")
        ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_dir)
        ocr_processor = TrOCRProcessor.from_pretrained(ocr_model_dir)
        
        # Set decoder_start_token_id
        ocr_model.config.decoder_start_token_id = ocr_processor.tokenizer.bos_token_id
        ocr_model.config.pad_token_id = ocr_processor.tokenizer.pad_token_id
        ocr_model.config.eos_token_id = ocr_processor.tokenizer.eos_token_id
    
    # Adapt tokenizer for Vietnamese
    # Add missing Vietnamese characters to the vocabulary
    base_chars = "aăâbcdđeêghiklmnoôơpqrstuưvxyAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY"
    diacritics = "àáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴ"
    combinations = "ăắằẳẵặâấầẩẫậêếềểễệôốồổỗộơớờởỡợưứừửữựĂẮẰẲẴẶÂẤẦẨẪẬÊẾỀỂỄỆÔỐỒỔỖỘƠỚỜỞỠỢƯỨỪỬỮỰ"
    punctuation = ".,!?;:()[]{}-–—\"'""''…0123456789 "
    
    all_chars = base_chars + diacritics + combinations + punctuation
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    
    # Get the current vocabulary
    tokenizer = ocr_processor.tokenizer
    current_vocab = tokenizer.get_vocab()
    current_tokens = set(current_vocab.keys())
    
    # Add missing Vietnamese characters to the vocabulary
    new_tokens = []
    for char in all_chars:
        if char not in current_tokens and char not in special_tokens:
            new_tokens.append(char)
    
    if new_tokens:
        logger.info(f"Adding {len(new_tokens)} new tokens to the OCR vocabulary")
        tokenizer.add_tokens(new_tokens)
        
        # Resize token embeddings of the decoder model
        ocr_model.decoder.resize_token_embeddings(len(tokenizer))

    # Train OCR model for Stage 1
    stage1_results = train_ocr_stage1(
        model=ocr_model,
        processor=ocr_processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=os.path.join(output_dir, "stage1"),
        batch_size=ocr_batch_size,
        learning_rate=ocr_lr,
        num_epochs=ocr_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        device=device,
        max_grad_norm=max_grad_norm,
        fp16=fp16
    )
    
    # Load best OCR model
    best_ocr_path = stage1_results["best_model_path"]
    logger.info(f"Loading best OCR model from {best_ocr_path}")
    ocr_model = VisionEncoderDecoderModel.from_pretrained(best_ocr_path)
    ocr_processor = TrOCRProcessor.from_pretrained(best_ocr_path)
    
    # Stage 2: Train accent model for diacritic insertion
    logger.info("=== Stage 2: Training accent model for diacritic insertion ===")
    
    if accent_model_dir and os.path.exists(accent_model_dir):
        # Load pretrained accent model
        logger.info(f"Loading pretrained accent model from {accent_model_dir}")
        accent_model = VietnameseAccentClassifier.from_pretrained(accent_model_dir)
        accent_processor = VietnameseAccentProcessor.from_pretrained(accent_model_dir)
    else:
        # Train accent model
        stage2_results = train_accent_stage2(
            corpus_file=corpus_file,
            mapping_file=mapping_file,
            output_dir=os.path.join(output_dir, "stage2"),
            pretrained_model="xlm-roberta-base",
            batch_size=accent_batch_size,
            learning_rate=accent_lr,
            num_epochs=accent_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            device=device,
            max_grad_norm=max_grad_norm,
            fp16=fp16,
            max_length=max_length,
            split_ratio=0.9
        )
        
        # Load best accent model
        accent_model = stage2_results["model"]
        accent_processor = stage2_results["processor"]
    
    # Create integrated model
    integrated_model = TwoStageVietnameseOCR(
        ocr_model=ocr_model,
        ocr_processor=ocr_processor,
        accent_model=accent_model,
        accent_processor=accent_processor,
        stage=2  # Full OCR + accent
    )
    
    # Save integrated model
    integrated_model.save_pretrained(os.path.join(output_dir, "final"))
    logger.info(f"Saved integrated model to {os.path.join(output_dir, 'final')}")
    
    # Return integrated model
    return integrated_model


# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Vietnamese OCR model with two stages")
    
    # Data arguments
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to training dataset")
    parser.add_argument("--val_dataset", type=str, required=True,
                        help="Path to validation dataset")
    parser.add_argument("--corpus_file", type=str, required=True,
                        help="Path to Vietnamese corpus with diacritics")
    parser.add_argument("--mapping_file", type=str, required=True,
                        help="Path to accent mapping file")
    parser.add_argument("--output_dir", type=str, default="./vietnamese-ocr-model",
                        help="Directory to save the model")
    
    # Model arguments
    parser.add_argument("--ocr_model", type=str, default="microsoft/trocr-base-handwritten",
                        help="Pretrained OCR model to start from")
    parser.add_argument("--accent_model", type=str, default=None,
                        help="Pretrained accent model or None to train from scratch")
    
    # Training arguments
    parser.add_argument("--ocr_batch_size", type=int, default=8,
                        help="Batch size for OCR training")
    parser.add_argument("--accent_batch_size", type=int, default=8,
                        help="Batch size for accent training")
    parser.add_argument("--ocr_lr", type=float, default=5e-5,
                        help="Learning rate for OCR model")
    parser.add_argument("--accent_lr", type=float, default=2e-5,
                        help="Learning rate for accent model")
    parser.add_argument("--ocr_epochs", type=int, default=3,
                        help="Number of epochs for OCR training")
    parser.add_argument("--accent_epochs", type=int, default=3,
                        help="Number of epochs for accent training")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load OCR datasets
    from datasets import load_dataset, concatenate_datasets, DatasetDict
    
    logger.info("Loading OCR datasets...")
    dataset2 = load_dataset('vklinhhh/4_cinnamon_ai_overlay_background_compressed')
    dataset3 = load_dataset('vklinhhh/5_uit_line')
    
    # Combine training sets and create a validation split
    combined_train = concatenate_datasets([dataset2['train'], dataset3['train']])
    # combined_train = dataset2['train']
    # Apply split
    validation_split = 0.1
    train_val_split = combined_train.train_test_split(test_size=validation_split, seed=args.seed)
    
    # Create the DatasetDict with train and validation splits
    word_dataset = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test']  # validation set
    })
    
    # Train the integrated model
    integrated_model = train_integrated_model(
        ocr_model_dir=args.ocr_model,
        accent_model_dir=args.accent_model,
        train_dataset=word_dataset["train"],
        val_dataset=word_dataset["validation"],
        corpus_file=args.corpus_file,
        mapping_file=args.mapping_file,
        output_dir=args.output_dir,
        ocr_batch_size=args.ocr_batch_size,
        accent_batch_size=args.accent_batch_size,
        ocr_lr=args.ocr_lr,
        accent_lr=args.accent_lr,
        ocr_epochs=args.ocr_epochs,
        accent_epochs=args.accent_epochs,
        fp16=args.fp16
    )
    
    logger.info("Training complete!")