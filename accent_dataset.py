import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
from utils import strip_vietnamese_diacritics


class VietnameseAccentDataset(Dataset):
    """
    Dataset for Vietnamese accent insertion task.
    Takes accented Vietnamese text, removes diacritics, and trains model
    to predict the original accent marks.
    """
    def __init__(
        self,
        texts,  # List of sentences with full diacritics
        accent_label_map,  # Mapping from unaccented to accent class
        accent_base_word_map,  # Add this parameter
        tokenizer,
        max_length=128,
        is_training=True,
        unaccented_texts=None  # For inference
    ):
        self.texts = texts
        self.accent_label_map = accent_label_map
        self.accent_base_word_map = accent_base_word_map  # Store the base word map
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # If unaccented texts are provided (for inference), use them
        # Otherwise, strip diacritics from the accented texts
        if unaccented_texts is not None:
            self.unaccented_texts = unaccented_texts
        else:
            self.unaccented_texts = [strip_vietnamese_diacritics(text) for text in texts]
    
    def __len__(self):
        return len(self.unaccented_texts)
    
    def __getitem__(self, idx):
        unaccented_text = self.unaccented_texts[idx]
        
        # For training, also provide accented text
        if self.is_training:
            accented_text = self.texts[idx]
            encoding = self.processor.prepare_accent_data(unaccented_text, accented_text)
        else:
            # For inference, only provide unaccented text
            encoding = self.processor.prepare_accent_data(unaccented_text)
        
        # Convert to tensors and return
        result = {}
        for key, value in encoding.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)  # Remove batch dimension
        
        # Store original texts for reference
        result["unaccented_text"] = unaccented_text
        if self.is_training:
            result["accented_text"] = accented_text
            
        return result


class VietnameseAccentTaggingDataset(Dataset):
    """
    Dataset for Vietnamese accent tagging using the dedicated mapping file.
    This version creates training examples at the word level.
    """
    def __init__(
        self,
        texts,  # List of sentences with full diacritics
        accent_label_map,  # Mapping from unaccented to accent class
        accent_base_word_map,  # Add this parameter
        tokenizer,
        max_length=128,
        is_training=True,
        unaccented_texts=None  # For inference
    ):
        self.texts = texts
        self.accent_label_map = accent_label_map
        self.accent_base_word_map = accent_base_word_map  # Store the base word map
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # For inference, use provided unaccented texts
        if unaccented_texts is not None:
            self.unaccented_texts = unaccented_texts
        else:
            # For training, strip diacritics
            self.unaccented_texts = [strip_vietnamese_diacritics(text) for text in texts]
        
        # Pre-process sentences into words for tagging
        self.examples = self._prepare_examples()
    def _prepare_examples(self):
        """Prepare word-level examples for accent tagging"""
        examples = []
        
        # Add debug counter
        skipped_word_count_mismatch = 0
        skipped_no_labels = 0
        total_words = 0
        words_with_labels = 0
        
        for i, (unaccented_text, accented_text) in enumerate(zip(self.unaccented_texts, self.texts)):
            # Split into words
            unaccented_words = unaccented_text.split()
            accented_words = accented_text.split()
            
            # Ensure same number of words (should be the case with proper stripping)
            if len(unaccented_words) != len(accented_words):
                print(f"Warning: Word count mismatch in example {i}. Skipping.")
                print(f"Unaccented: {unaccented_words}")
                print(f"Accented: {accented_words}")
                skipped_word_count_mismatch += 1
                continue
            
            # Create example
            example = {
                "sentence_idx": i,
                "unaccented_words": unaccented_words,
                "accented_words": accented_words,
                "accent_labels": []
            }
            
            # Create word-level labels
            valid_labels_found = False
            for j, (unaccented, accented) in enumerate(zip(unaccented_words, accented_words)):
                total_words += 1
                
                # Normalize word pair to handle hyphenation and punctuation
                unaccented_normalized = re.sub(r'[^\w\s]', '', unaccented.lower())
                accented_normalized = re.sub(r'[^\w\s]', '', accented.lower())
                
                # Get label ID for this word
                label_id = -100  # Default: ignore index
                
                # NEW APPROACH: Check if the base word exists in our mapping
                if unaccented_normalized in self.accent_base_word_map:
                    # Check if this exact accented form exists for this base word
                    if accented_normalized in self.accent_base_word_map[unaccented_normalized]:
                        label_id = self.accent_base_word_map[unaccented_normalized][accented_normalized]
                        valid_labels_found = True
                        words_with_labels += 1
                
                example["accent_labels"].append(label_id)
            
            # Don't add examples with no labels
            if not valid_labels_found:
                skipped_no_labels += 1
                continue
                    
            examples.append(example)
        
        print(f"Created {len(examples)} examples for training/validation")
        print(f"Skipped {skipped_word_count_mismatch} examples due to word count mismatch")
        print(f"Skipped {skipped_no_labels} examples due to no valid labels")
        print(f"Total words processed: {total_words}, words with valid labels: {words_with_labels} ({words_with_labels/max(1,total_words)*100:.2f}%)")
        
        # Print a sample of the first example if available
        if examples:
            sample_example = examples[0]
            print("\nSample example:")
            print(f"Unaccented words: {sample_example['unaccented_words'][:10]}")
            print(f"Accented words: {sample_example['accented_words'][:10]}")
            print(f"Accent labels: {sample_example['accent_labels'][:10]}")
        
        return examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Get the original sentence index
        sentence_idx = example["sentence_idx"]
        
        # Get the full unaccented sentence
        unaccented_text = self.unaccented_texts[sentence_idx]
        
        # Tokenize the input text
        encoding = self.tokenizer(
            unaccented_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create result dict
        result = {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "unaccented_text": unaccented_text
        }
        
        # For training, add accent labels and original text
        if self.is_training:
            # Create label tensor (all -100 initially)
            labels = torch.full_like(encoding.input_ids.squeeze(0), -100)
            
            # Map words to token positions (simplified approach)
            token_ids = encoding.input_ids.squeeze(0).tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Track word position
            current_word_idx = -1
            current_word = ""
            
            # Assign labels to tokens
            for i, token in enumerate(tokens):
                # Skip special tokens
                if token in self.tokenizer.all_special_tokens:
                    continue
                
                # Check if this is a new word
                if token.startswith("▁"):  # XLM-RoBERTa uses ▁ to mark word beginnings
                    current_word_idx += 1
                    current_word = token[1:]  # Remove the ▁
                else:
                    current_word += token  # Continue building the subword
                
                # Check if we're still within word range and the label exists
                if 0 <= current_word_idx < len(example["accent_labels"]):
                    word_label = example["accent_labels"][current_word_idx]
                    if word_label != -100:
                        labels[i] = word_label
            
            result["accent_labels"] = labels
            result["accented_text"] = self.texts[sentence_idx]
        
        return result


class TokenizedDataset(torch.utils.data.Dataset):
    """Dataset for pre-tokenized Vietnamese accent data"""
    
    def __init__(self, tokenized_file):
        """
        Args:
            tokenized_file: Path to pre-tokenized data (.pt file)
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading tokenized data from {tokenized_file}")
        self.examples = torch.load(tokenized_file)
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Return tensors needed for training
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "accent_labels": example["accent_labels"]
        }


def build_accent_dataloaders(
    texts,
    processor,
    batch_size=16,
    max_length=128,
    is_training=True,
    unaccented_texts=None,
    num_workers=4,
    use_tagging_dataset=True
):
    """
    Build DataLoader for accent insertion task.
    
    Args:
        texts: List of Vietnamese texts with diacritics
        processor: Processor for accent prediction
        batch_size: Batch size
        max_length: Maximum sequence length
        is_training: Whether this is for training
        unaccented_texts: Optional list of texts without diacritics (for inference)
        num_workers: Number of workers for data loading
        use_tagging_dataset: Whether to use tagging dataset (word-level) or sequence dataset
        
    Returns:
        DataLoader for accent training/inference
    """

    if use_tagging_dataset:
        # Use tagging dataset with accent map
        dataset = VietnameseAccentTaggingDataset(
            texts=texts,
            accent_label_map=processor.accent_label_map,
            accent_base_word_map=processor.accent_base_word_map,  # Add this parameter
            tokenizer=processor.tokenizer,
            max_length=max_length,
            is_training=is_training,
            unaccented_texts=unaccented_texts
        )
    else:
        # Use sequence-level dataset
        dataset = VietnameseAccentDataset(
            texts=texts,
            processor=processor,
            max_length=max_length,
            is_training=is_training,
            unaccented_texts=unaccented_texts
        )
    
    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True
    )


def prepare_corpus_dataset(corpus_file_path, split_ratio=0.9):
    """
    Prepare dataset from a corpus file.
    
    Args:
        corpus_file_path: Path to corpus file
        split_ratio: Train/val split ratio
        
    Returns:
        Dict with train and val texts
    """
    # Read corpus file
    with open(corpus_file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Shuffle texts
    import random
    random.shuffle(texts)
    
    # Split into train and val
    split_idx = int(len(texts) * split_ratio)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    return {
        "train": train_texts,
        "val": val_texts
    }