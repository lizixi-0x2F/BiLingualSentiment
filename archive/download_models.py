#!/usr/bin/env python
"""
Download pre-trained models for bilingual sentiment analysis.

This script downloads the recommended pre-trained models:
- XLM-RoBERTa Base: excellent for cross-lingual transfer
- DistilBERT Multilingual: lighter and faster alternative

The models are saved to the pretrained_models directory.
"""

import os
import argparse
import logging
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def download_model(model_name, output_dir, force=False):
    """
    Download model and tokenizer from Hugging Face.
    
    Args:
        model_name: Name of the model on Hugging Face
        output_dir: Directory to save the model
        force: If True, download even if model already exists
    """
    output_path = Path(output_dir)
    
    # Check if model already exists
    model_config_path = output_path / "config.json"
    if model_config_path.exists() and not force:
        logger.info(f"Model {model_name} already exists at {output_dir}. Use --force to redownload.")
        return
    
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name}...")
    start_time = time.time()
    
    try:
        # Download model
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save model and tokenizer
        logger.info(f"Saving model and tokenizer to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully downloaded {model_name} to {output_dir} in {elapsed_time:.2f} seconds")
        
        # Get model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        logger.info(f"Model size: {model_size:.2f} MB")
        
        # Test if model works
        test_model_loading(output_dir)
        
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        raise

def test_model_loading(model_dir):
    """
    Test if we can load the model from the saved directory.
    
    Args:
        model_dir: Directory where model was saved
    """
    try:
        logger.info(f"Testing if model can be loaded from {model_dir}...")
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Create a simple input to test the model
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info(f"Model loaded and tested successfully from {model_dir}")
    except Exception as e:
        logger.error(f"Error loading model from {model_dir}: {e}")

def main():
    """Main function to parse arguments and download models."""
    parser = argparse.ArgumentParser(description="Download pre-trained models for bilingual sentiment analysis")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["xlm-roberta-base", "distilbert-base-multilingual-cased", "all"],
        default=["all"],
        help="Which models to download"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force download even if models already exist"
    )
    args = parser.parse_args()
    
    # Define models to download
    all_models = {
        "xlm-roberta-base": "pretrained_models/xlm-roberta-base",
        "distilbert-base-multilingual-cased": "pretrained_models/distilbert-multilingual"
    }
    
    # Select models to download based on arguments
    models_to_download = all_models
    if "all" not in args.models:
        models_to_download = {k: all_models[k] for k in args.models}
    
    # Print summary
    logger.info(f"Will download {len(models_to_download)} models to pretrained_models/")
    for model_name, output_dir in models_to_download.items():
        logger.info(f"  - {model_name} → {output_dir}")
    
    # Download each model
    for model_name, output_dir in models_to_download.items():
        download_model(model_name, output_dir, args.force)
    
    logger.info("All specified models downloaded successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Update your training script to use local models:")
    logger.info("   model_path = 'pretrained_models/xlm-roberta-base'  # or distilbert-multilingual")
    logger.info("2. Train your model using train.py or train_pretrained.py")

if __name__ == "__main__":
    main()
