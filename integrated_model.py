import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel
from typing import Dict, List, Optional, Tuple, Union, Any

from accent_model import VietnameseAccentClassifier, VietnameseAccentProcessor
from utils import strip_vietnamese_diacritics


class TwoStageVietnameseOCR(nn.Module):
    """
    Integrated two-stage model for Vietnamese OCR:
    1. TrOCR for text recognition without diacritics
    2. XLM-RoBERTa for accent insertion
    """
    def __init__(
        self,
        ocr_model,
        ocr_processor,
        accent_model=None,
        accent_processor=None,
        stage=1  # 1: OCR only, 2: OCR + Accent
    ):
        super(TwoStageVietnameseOCR, self).__init__()
        self.ocr_model = ocr_model
        self.ocr_processor = ocr_processor
        self.accent_model = accent_model
        self.accent_processor = accent_processor
        self.stage = stage
    
    def set_stage(self, stage):
        """Set the current stage (1 or 2)"""
        assert stage in [1, 2], "Stage must be 1 or 2"
        self.stage = stage
        return self
    
    def forward(
        self,
        pixel_values,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs
    ):
        """
        Forward pass through the model.
        Different behavior based on the current stage.
        """
        # Stage 1: Only OCR model
        if self.stage == 1 or self.accent_model is None:
            return self.ocr_model(
                pixel_values=pixel_values,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs
            )
        
        # Stage 2: OCR + Accent
        # First, get OCR output
        with torch.no_grad():
            ocr_outputs = self.ocr_model.generate(
                pixel_values=pixel_values,
                max_length=128,
                num_beams=5,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Process OCR sequences
        ocr_sequences = ocr_outputs.sequences
        batch_size = ocr_sequences.shape[0]
        
        # Decode OCR outputs to get unaccented text
        unaccented_texts = []
        for i in range(batch_size):
            seq = ocr_sequences[i]
            text = self.ocr_processor.tokenizer.decode(seq, skip_special_tokens=True)
            unaccented_texts.append(text)
        
        # Process through accent model
        accent_inputs = []
        for text in unaccented_texts:
            # Prepare input for accent model
            accent_input = self.accent_processor.prepare_accent_data(text)
            accent_inputs.append(accent_input)
        
        # Batch accent inputs
        batched_accent_input = {
            "input_ids": torch.cat([x["input_ids"] for x in accent_inputs]),
            "attention_mask": torch.cat([x["attention_mask"] for x in accent_inputs])
        }
        
        # Run accent model
        accent_outputs = self.accent_model(**batched_accent_input)
        
        # Overall loss combines OCR and accent losses if labels provided
        loss = None
        if labels is not None and "accent_labels" in kwargs:
            # Get OCR loss
            ocr_outputs = self.ocr_model(
                pixel_values=pixel_values,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs
            )
            ocr_loss = ocr_outputs.loss
            
            # Get accent loss
            accent_loss = accent_outputs["loss"] if "loss" in accent_outputs else None
            
            # Combine losses
            if accent_loss is not None:
                loss = ocr_loss + accent_loss
            else:
                loss = ocr_loss
        
        # Return combined results
        return {
            "loss": loss,
            "ocr_sequences": ocr_sequences,
            "unaccented_texts": unaccented_texts,
            "accent_logits": accent_outputs["logits"] if "logits" in accent_outputs else None
        }
    
    def generate(self, pixel_values, **kwargs):
        """
        Generate text from images.
        For stage 1, only OCR is used.
        For stage 2, OCR is followed by accent insertion.
        """
        # Stage 1: Only OCR
        if self.stage == 1 or self.accent_model is None:
            return self.ocr_model.generate(pixel_values=pixel_values, **kwargs)
        
        # Stage 2: OCR + Accent
        # First, run OCR model
        ocr_outputs = self.ocr_model.generate(
            pixel_values=pixel_values,
            max_length=128,
            num_beams=5,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Process OCR sequences
        ocr_sequences = ocr_outputs.sequences
        batch_size = ocr_sequences.shape[0]
        
        # Decode OCR outputs to get unaccented text
        unaccented_texts = []
        for i in range(batch_size):
            seq = ocr_sequences[i]
            text = self.ocr_processor.tokenizer.decode(seq, skip_special_tokens=True)
            unaccented_texts.append(text)
        
        # Apply accent model
        accented_texts = []
        for text in unaccented_texts:
            # Prepare input for accent model
            accent_input = self.accent_processor.prepare_accent_data(text)
            
            # Run accent model
            with torch.no_grad():
                accent_outputs = self.accent_model(
                    input_ids=accent_input["input_ids"].to(pixel_values.device),
                    attention_mask=accent_input["attention_mask"].to(pixel_values.device)
                )
            
            # Get predicted accent labels
            logits = accent_outputs["logits"] if isinstance(accent_outputs, dict) else accent_outputs
            predictions = self.accent_model.decode_accents(
                logits, 
                attention_mask=accent_input["attention_mask"].to(pixel_values.device)
            )
            
            # Apply accents to the unaccented text
            accented_text = self.accent_processor.apply_accents(text, predictions)
            accented_texts.append(accented_text)
        
        # Return accented texts
        return accented_texts
    
    def save_pretrained(self, output_dir):
        """Save the entire two-stage model"""
        import os
        import torch
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        ocr_dir = os.path.join(output_dir, "ocr_model")
        accent_dir = os.path.join(output_dir, "accent_model")
        os.makedirs(ocr_dir, exist_ok=True)
        os.makedirs(accent_dir, exist_ok=True)
        
        # Save OCR model and processor
        self.ocr_model.save_pretrained(ocr_dir)
        self.ocr_processor.save_pretrained(ocr_dir)
        
        # Save accent model and processor if they exist
        if self.accent_model is not None:
            self.accent_model.save_pretrained(accent_dir)
        
        if self.accent_processor is not None:
            self.accent_processor.save_pretrained(accent_dir)
        
        # Save stage information
        import json
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump({"stage": self.stage}, f)
    
    @classmethod
    def from_pretrained(cls, model_path):
        """Load the entire two-stage model"""
        import os
        import json
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        # Check for model directories
        ocr_dir = os.path.join(model_path, "ocr_model")
        accent_dir = os.path.join(model_path, "accent_model")
        
        # Load OCR model and processor
        ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_dir)
        ocr_processor = TrOCRProcessor.from_pretrained(ocr_dir)
        
        # Load accent model and processor if they exist
        accent_model = None
        accent_processor = None
        if os.path.exists(accent_dir):
            from accent_model import VietnameseAccentClassifier, VietnameseAccentProcessor
            accent_model = VietnameseAccentClassifier.from_pretrained(accent_dir)
            accent_processor = VietnameseAccentProcessor.from_pretrained(accent_dir)
        
        # Load stage information
        stage = 1
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                stage = config.get("stage", 1)
        
        # Create and return the model
        return cls(
            ocr_model=ocr_model,
            ocr_processor=ocr_processor,
            accent_model=accent_model,
            accent_processor=accent_processor,
            stage=stage
        )