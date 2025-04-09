import os
import argparse
import logging
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict

from train_pipeline import (
    train_ocr_stage1, 
    train_accent_stage2, 
    train_integrated_model, 
    set_seed
)
from accent_model import (
    VietnameseAccentClassifier, 
    VietnameseAccentProcessor, 
    create_accent_label_map
)
from integrated_model import TwoStageVietnameseOCR
from inference import evaluate_model, predict_single_image, predict_directory


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Vietnamese OCR with Accent Insertion")
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true",
                            help="Train the model")
    mode_group.add_argument("--evaluate", action="store_true",
                            help="Evaluate the model")
    mode_group.add_argument("--predict", action="store_true",
                            help="Run inference")
    
    # Training arguments
    train_group = parser.add_argument_group("Training Arguments")
    train_group.add_argument("--corpus_file", type=str,
                            help="Path to Vietnamese corpus with diacritics")
    train_group.add_argument("--mapping_file", type=str,
                            help="Path to accent mapping file")
    train_group.add_argument("--ocr_model", type=str, default="microsoft/trocr-base-handwritten",
                            help="Pretrained OCR model to start from")
    train_group.add_argument("--accent_model", type=str, default=None,
                            help="Pretrained accent model or None to train from scratch")
    train_group.add_argument("--ocr_batch_size", type=int, default=8,
                            help="Batch size for OCR training")
    train_group.add_argument("--accent_batch_size", type=int, default=16,
                            help="Batch size for accent training")
    train_group.add_argument("--ocr_lr", type=float, default=5e-5,
                            help="Learning rate for OCR model")
    train_group.add_argument("--accent_lr", type=float, default=2e-5,
                            help="Learning rate for accent model")
    train_group.add_argument("--ocr_epochs", type=int, default=1,
                            help="Number of epochs for OCR training")
    train_group.add_argument("--accent_epochs", type=int, default=1,
                            help="Number of epochs for accent training")
    train_group.add_argument("--train_stage", type=int, choices=[1, 2, 3], default=3,
                            help="1: OCR only, 2: Accent only, 3: Both")
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Arguments")
    eval_group.add_argument("--test_file", type=str,
                           help="Path to test file with image_path,ground_truth pairs")
    eval_group.add_argument("--stage", type=int, choices=[1, 2], default=2,
                           help="1: OCR only, 2: OCR + accent")
    
    # Prediction arguments
    pred_group = parser.add_argument_group("Prediction Arguments")
    pred_group.add_argument("--image", type=str,
                           help="Path to single image for prediction")
    pred_group.add_argument("--image_dir", type=str,
                           help="Directory with images for batch prediction")
    pred_group.add_argument("--extension", type=str, default=".jpg",
                           help="Image file extension for directory mode")
    
    # Common arguments
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory to save/load model")
    parser.add_argument("--output", type=str, default="output.txt",
                       help="Path to output file")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Training mode
    if args.train:
        if not args.corpus_file or not args.mapping_file:
            raise ValueError("Corpus file and mapping file required for training")
        
        logger.info(f"Starting training with stage {args.train_stage}")
        
        # Load OCR datasets
        if args.train_stage in [1, 3]:  # OCR or both
            logger.info("Loading OCR datasets...")
            try:
                dataset2 = load_dataset('vklinhhh/4_cinnamon_ai_overlay_background_compressed')
                # dataset3 = load_dataset('vklinhhh/5_uit_line')
                
                # Combine training sets and create a validation split
                # combined_train = concatenate_datasets([dataset2['train'], dataset3['train']])
                combined_train = dataset2['train']
                # Apply split
                validation_split = 0.1
                train_val_split = combined_train.train_test_split(test_size=validation_split, seed=args.seed)
                
                # Create the DatasetDict with train and validation splits
                word_dataset = DatasetDict({
                    'train': train_val_split['train'],
                    'validation': train_val_split['test']  # validation set
                })
            except Exception as e:
                logger.error(f"Error loading datasets: {e}")
                logger.info("Using local datasets if available...")
                
                # Try to load from local files
                if os.path.exists("train_dataset.json") and os.path.exists("val_dataset.json"):
                    import json
                    with open("train_dataset.json", 'r', encoding='utf-8') as f:
                        train_data = json.load(f)
                    with open("val_dataset.json", 'r', encoding='utf-8') as f:
                        val_data = json.load(f)
                    
                    word_dataset = {
                        'train': train_data,
                        'validation': val_data
                    }
                else:
                    raise ValueError("Could not load datasets and no local datasets found")
        
        # Stage 1: OCR only
        if args.train_stage == 1:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Load or create OCR model
            try:
                # Try to load pretrained model
                ocr_model = VisionEncoderDecoderModel.from_pretrained(args.ocr_model)
                ocr_processor = TrOCRProcessor.from_pretrained(args.ocr_model)
                logger.info(f"Loaded pretrained OCR model from {args.ocr_model}")
            except:
                # Create new model
                logger.info(f"Creating new OCR model from {args.ocr_model}")
                ocr_model = VisionEncoderDecoderModel.from_pretrained(args.ocr_model)
                ocr_processor = TrOCRProcessor.from_pretrained(args.ocr_model)
                
                # Set decoder_start_token_id
                ocr_model.config.decoder_start_token_id = ocr_processor.tokenizer.bos_token_id
                ocr_model.config.pad_token_id = ocr_processor.tokenizer.pad_token_id
                ocr_model.config.eos_token_id = ocr_processor.tokenizer.eos_token_id
            
            # Train OCR model
            results = train_ocr_stage1(
                model=ocr_model,
                processor=ocr_processor,
                train_dataset=word_dataset["train"],
                val_dataset=word_dataset["validation"],
                output_dir=args.model_dir,
                batch_size=args.ocr_batch_size,
                learning_rate=args.ocr_lr,
                num_epochs=args.ocr_epochs,
                fp16=args.fp16
            )
            
            logger.info(f"OCR training complete. Best model path: {results['best_model_path']}")
            
        # Stage 2: Accent only
        elif args.train_stage == 2:
            # Train accent model
            results = train_accent_stage2(
                corpus_file=args.corpus_file,
                mapping_file=args.mapping_file,
                output_dir=args.model_dir,
                batch_size=args.accent_batch_size,
                learning_rate=args.accent_lr,
                num_epochs=args.accent_epochs,
                fp16=args.fp16
            )
            
            logger.info(f"Accent model training complete. Best model path: {results['best_model_path']}")
            
        # Stage 3: Integrated training
        else:
            # Train integrated model
            integrated_model = train_integrated_model(
                ocr_model_dir=args.ocr_model,
                accent_model_dir=args.accent_model,
                train_dataset=word_dataset["train"],
                val_dataset=word_dataset["validation"],
                corpus_file=args.corpus_file,
                mapping_file=args.mapping_file,
                output_dir=args.model_dir,
                ocr_batch_size=args.ocr_batch_size,
                accent_batch_size=args.accent_batch_size,
                ocr_lr=args.ocr_lr,
                accent_lr=args.accent_lr,
                ocr_epochs=args.ocr_epochs,
                accent_epochs=args.accent_epochs,
                fp16=args.fp16
            )

            logger.info(f"Integrated model training complete. Model saved to {args.model_dir}")
    
    # Evaluation mode
    elif args.evaluate:
        if not args.test_file:
            raise ValueError("Test file required for evaluation")
        
        # Load model
        logger.info(f"Loading model from {args.model_dir}")
        model = TwoStageVietnameseOCR.from_pretrained(args.model_dir)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Load test data
        test_data = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_path = parts[0]
                    ground_truth = parts[1]
                    test_data.append((image_path, ground_truth))
        
        logger.info(f"Loaded {len(test_data)} test examples")
        
        # Evaluate model
        metrics = evaluate_model(model, test_data, args.output, args.stage)
        
        # Print results
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
    
    # Prediction mode
    elif args.predict:
        # Load model
        logger.info(f"Loading model from {args.model_dir}")
        model = TwoStageVietnameseOCR.from_pretrained(args.model_dir)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Set model stage
        model.set_stage(args.stage)
        
        # Single image prediction
        if args.image:
            logger.info(f"Predicting text from {args.image}")
            prediction = predict_single_image(model, args.image, args.stage)
            logger.info(f"Prediction: {prediction}")
            
            # Save to output file
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Image Path\tPrediction\n")
                f.write(f"{args.image}\t{prediction}\n")
            
            logger.info(f"Saved prediction to {args.output}")
            
        # Directory prediction
        elif args.image_dir:
            logger.info(f"Predicting text from images in {args.image_dir}")
            predict_directory(model, args.image_dir, args.output, args.stage, args.extension)
            
        else:
            raise ValueError("Either --image or --image_dir must be specified for prediction")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()