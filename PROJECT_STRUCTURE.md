# Bilingual Sentiment Analysis Project Structure

This file documents the organized project structure after cleanup.

## Project Root
- `README.md` - Project documentation
- `IMPLEMENTATION_NOTES.md` - Implementation details and future plans
- `PROJECT_STRUCTURE.md` - This file
- `requirements.txt` - Python dependencies
- `train.py` - Main training script
- `train_pretrained.py` - Training script for pre-trained models
- `download_models.py` - Script to download pre-trained models
- `Chinese_VA_dataset_gaussNoise.csv` - Chinese dataset with valence-arousal labels
- `emobank_va_normalized.csv` - English dataset with valence-arousal labels
- `example_texts.txt` - Example texts for testing

## Core Directories
- `src/` - Source code modules
  - `models/` - Model implementations
  - `utils/` - Utility functions
  - `config.py` - Configuration parameters
  - `inference.py` - Inference script for trained models
- `shells/` - Shell scripts for running training and inference
- `pretrained_models/` - Downloaded pre-trained models
  - `distilbert-multilingual/` - DistilBERT multilingual model files
  - `xlm-roberta-base/` - XLM-RoBERTa model files
- `outputs/` - Model outputs and visualizations
  - `pretrained_distilbert_local/` - DistilBERT fine-tuned model
  - `pretrained_xlm_roberta_local/` - XLM-RoBERTa fine-tuned model

## Additional Directories
- `examples/` - Example implementations and usage
  - `train_with_amp.py` - Example of training with automatic mixed precision
- `archive/` - Archived development files
  - `benchmark_results_final/` - Benchmark results
  - `benchmark_results_test/` - Test benchmark results
  - Various optimization and visualization scripts
- `cleanup_backup/` - Backup of removed files during cleanup

## Key Files
### Source Code
- `src/models/roberta_model.py` - Implementation of the transformer-based models
- `src/utils/data_utils.py` - Dataset and data loading utilities
- `src/utils/train_utils.py` - Training utilities
- `src/utils/visualization.py` - Visualization tools

### Shell Scripts
- `shells/run_train_local_models.ps1` - PowerShell script to train models
- `shells/run_train_local_models.sh` - Bash script to train models
- `shells/test_model.ps1` - PowerShell script to test models
- `shells/test_model.sh` - Bash script to test models

### Outputs
- `outputs/*/best_model.pth` - Best model weights
- `outputs/*/test_results.json` - Test results
- `outputs/*/metrics_epoch_*.json` - Per-epoch metrics
- `outputs/*/visualizations/` - Visualization outputs
