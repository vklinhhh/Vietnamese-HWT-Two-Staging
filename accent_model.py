import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizer
from typing import Dict, List, Optional, Tuple, Union


class VietnameseAccentClassifier(nn.Module):
    """
    Model for inserting Vietnamese diacritics based on XLM-RoBERTa.
    Takes unaccented Vietnamese text and predicts the correct accented version.
    """
    def __init__(
        self,
        pretrained_model_name="xlm-roberta-base",
        num_accent_labels=None,
        accent_label_map=None,
        dropout_rate=0.1,
        use_crf=False
    ):
        super(VietnameseAccentClassifier, self).__init__()
        
        # Load XLM-RoBERTa as the base model
        self.config = XLMRobertaConfig.from_pretrained(pretrained_model_name)
        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name)
        
        # Store accent label mapping
        self.accent_label_map = accent_label_map
        
        # Determine number of accent classes
        if num_accent_labels is None and accent_label_map is not None:
            self.num_accent_labels = len(accent_label_map)
        else:
            self.num_accent_labels = num_accent_labels
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head for accent prediction
        self.accent_classifier = nn.Linear(self.config.hidden_size, self.num_accent_labels)
        
        # Optional CRF layer for sequence labeling
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(self.num_accent_labels, batch_first=True)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        accent_labels=None,
        return_dict=True
    ):
        """
        Forward pass for accent classification.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            token_type_ids: Token type IDs (not used for XLM-RoBERTa)
            accent_labels: Ground truth accent labels (optional)
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dict with loss and logits or just logits if no labels provided
        """
        # Get contextualized embeddings from RoBERTa
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Predict accent logits
        logits = self.accent_classifier(sequence_output)
        
        loss = None
        if accent_labels is not None:
            if self.use_crf:
                # Use CRF for sequence labeling
                # CRF expects labels as long tensors
                log_likelihood = self.crf(logits, accent_labels, mask=attention_mask.bool(), reduction='mean')
                loss = -log_likelihood
            else:
                # Standard cross-entropy loss for classification
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # Only consider non-padding tokens in loss calculation
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_accent_labels)
                active_labels = torch.where(
                    active_loss, accent_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(accent_labels)
                )
                loss = loss_fct(active_logits, active_labels)
        
        if return_dict:
            return {
                "loss": loss,
                "logits": logits
            }
        else:
            return (loss, logits) if loss is not None else logits
    
    def decode_accents(self, logits, attention_mask=None):
        """
        Decode accent logits to predicted accent labels.
        
        Args:
            logits: Model output logits
            attention_mask: Attention mask for padding
            
        Returns:
            List of predicted accent labels
        """
        if self.use_crf:
            # Use CRF for decoding
            # Convert attention_mask to boolean mask for CRF
            mask = attention_mask.bool() if attention_mask is not None else None
            predictions = self.crf.decode(logits, mask=mask)
            return predictions
        else:
            # Standard argmax for classification
            predictions = torch.argmax(logits, dim=-1)
            return predictions.cpu().numpy()


    def save_pretrained(self, output_dir):
        """Save model to directory"""
        # Create the output directory and any parent directories
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model weights
        torch.save(self.state_dict(), f"{output_dir}/accent_model.pt")
        
        # Save the config
        self.config.save_pretrained(output_dir)
        
        # Save accent label map if it exists
        if self.accent_label_map is not None:
            import json
            with open(f"{output_dir}/accent_label_map.json", 'w', encoding='utf-8') as f:
                json.dump(self.accent_label_map, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, model_path):
        """Load model from directory"""
        import json
        import os
        
        # Load config
        config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")  # Always use the original model ID
        
        # Load accent label map if it exists
        accent_label_map = None
        accent_map_path = os.path.join(model_path, "accent_label_map.json")
        if os.path.exists(accent_map_path):
            with open(accent_map_path, 'r', encoding='utf-8') as f:
                accent_label_map = json.load(f)
        
        # Create model instance
        model = cls(
            pretrained_model_name="xlm-roberta-base",  # Use the original model ID
            num_accent_labels=len(accent_label_map) if accent_label_map else None,
            accent_label_map=accent_label_map
        )
        
        # Load model weights
        model_weights_path = os.path.join(model_path, "accent_model.pt")
        if os.path.exists(model_weights_path):
            model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
        else:
            raise ValueError(f"Model weights file 'accent_model.pt' not found in {model_path}")
        
        return model

class VietnameseAccentProcessor:
    def __init__(
        self,
        tokenizer_name="xlm-roberta-base",
        accent_label_map=None,
        accent_base_word_map=None, 
        max_length=128
    ):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        self.accent_label_map = accent_label_map
        self.accent_base_word_map = accent_base_word_map  # Store the base word map
        self.max_length = max_length
        
        # Create reverse mapping from label to accented word
        if accent_label_map:
            self.reverse_map = {idx: accented for accented, idx in accent_label_map.items()}

    def save_pretrained(self, output_dir):
        """Save processor to directory"""
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save accent label map if it exists
        if self.accent_label_map is not None:
            import json
            with open(f"{output_dir}/accent_label_map.json", 'w', encoding='utf-8') as f:
                json.dump(self.accent_label_map, f, ensure_ascii=False, indent=2)
                
        # Save base word map if it exists
        if hasattr(self, 'accent_base_word_map') and self.accent_base_word_map is not None:
            import json
            with open(f"{output_dir}/accent_base_word_map.json", 'w', encoding='utf-8') as f:
                json.dump(self.accent_base_word_map, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path):
        """Load processor from directory"""
        import json
        import os
        
        # Load accent label map if it exists
        accent_label_map = None
        accent_map_path = os.path.join(model_path, "accent_label_map.json")
        if os.path.exists(accent_map_path):
            with open(accent_map_path, 'r', encoding='utf-8') as f:
                accent_label_map = json.load(f)
        
        # Load base word map if it exists
        base_word_map = None
        base_word_map_path = os.path.join(model_path, "accent_base_word_map.json")
        if os.path.exists(base_word_map_path):
            with open(base_word_map_path, 'r', encoding='utf-8') as f:
                base_word_map = json.load(f)
        
        return cls(
            tokenizer_name=model_path,
            accent_label_map=accent_label_map,
            accent_base_word_map=base_word_map
        )
    
    def prepare_accent_data(self, unaccented_text, accented_text=None):
        """
        Prepare data for accent model training or inference.
        
        Args:
            unaccented_text: Text without diacritics
            accented_text: Target text with diacritics (optional for training)
            
        Returns:
            Dict with model inputs and optionally labels
        """
        # Tokenize input text
        encoding = self.tokenizer(
            unaccented_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask
        }
        
        # If accented text is provided, prepare labels
        if accented_text is not None and self.accent_label_map is not None:
            # Tokenize without padding to get token alignment
            unaccented_tokens = self.tokenizer.tokenize(unaccented_text)
            
            # Create labels tensor initialized to ignore index (-100)
            labels = torch.full_like(encoding.input_ids, -100)
            
            # Map tokens to their accent labels
            # This is a simplified approach and would need more work for subword tokenization
            for i, token in enumerate(unaccented_tokens):
                if i < self.max_length - 2:  # Account for special tokens
                    # Find corresponding token in accented text
                    # (This is a simplification - would need proper alignment in practice)
                    if token in self.accent_label_map:
                        labels[0, i+1] = self.accent_label_map[token]  # +1 for <s> token
            
            result["accent_labels"] = labels
        
        return result
    
    def apply_accents(self, unaccented_text, accent_predictions):
        """
        Apply predicted accent labels to unaccented text.
        
        Args:
            unaccented_text: Original text without diacritics
            accent_predictions: Predicted accent labels from model
            
        Returns:
            Text with diacritics applied
        """
        import numpy as np
        if self.accent_label_map is None:
            return unaccented_text
        
        # Check if accent_predictions is a 2D array (batch x sequence)
        if isinstance(accent_predictions, np.ndarray) and accent_predictions.ndim > 1:
            # Take the first sequence in the batch
            accent_predictions = accent_predictions[0]
        
        # Tokenize to align with predictions
        tokens = self.tokenizer.tokenize(unaccented_text)
        
        # Convert tokens based on predictions
        accented_tokens = []
        for i, token in enumerate(tokens):
            if i < len(accent_predictions):
                # Extract the prediction for this position, handling various formats
                if isinstance(accent_predictions, list):
                    # Handle list predictions (from CRF)
                    if i < len(accent_predictions):
                        pred_label = accent_predictions[i]
                    else:
                        pred_label = -100
                elif isinstance(accent_predictions, np.ndarray):
                    # Handle numpy array predictions
                    pred_label = int(accent_predictions[i])
                else:
                    # Other formats
                    pred_label = -100
                
                # Skip special tokens and invalid predictions
                if pred_label == -100 or (isinstance(pred_label, (int, float)) and pred_label >= len(self.reverse_map)):
                    accented_tokens.append(token)
                else:
                    # Map prediction to accented form
                    accented_form = self.reverse_map.get(pred_label, token)
                    accented_tokens.append(accented_form)
            else:
                accented_tokens.append(token)
        
        # Convert tokens back to text
        accented_text = self.tokenizer.convert_tokens_to_string(accented_tokens)
        return accented_text


def create_accent_label_map(mapping_file_path):
    """
    Create mapping from unaccented to accented Vietnamese text.
    
    Args:
        mapping_file_path: Path to file with mapping (format: unaccented-accented)
        
    Returns:
        Dict mapping unaccented-accented pairs to label indices
    """
    accent_map = {}
    label_to_accented = {}
    
    # Also create a map of base words to a dict of their accented forms
    base_word_map = {}
    
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    valid_lines = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            parts = line.split('-')
            if len(parts) == 2:
                unaccented, accented = parts
                # Store the full pattern as the key
                accent_map[f"{unaccented}-{accented}"] = i
                label_to_accented[i] = accented
                
                # Also store the base word as a key in the base_word_map
                if unaccented not in base_word_map:
                    base_word_map[unaccented] = {}
                base_word_map[unaccented][accented] = i
                
                valid_lines += 1
    
    print(f"Read {len(lines)} lines from mapping file, found {valid_lines} valid mappings")
    print(f"First few mappings: {list(accent_map.items())[:5]}")
    print(f"Found {len(base_word_map)} unique base words")
    
    return accent_map, label_to_accented, base_word_map 