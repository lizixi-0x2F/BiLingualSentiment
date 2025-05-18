# GitHub & Hugging Face Workflow

This document outlines the complete workflow for managing this project with GitHub and Hugging Face.

## Project Configuration

The project has been configured to:

1. **Store code on GitHub**: 
   - All code, configuration, and small files are stored on GitHub
   - Large model files (*.safetensors, *.bin) are excluded via .gitignore

2. **Store models on Hugging Face**:
   - Trained models are uploaded to Hugging Face
   - Scripts are provided for uploading and downloading models
   - This avoids GitHub's file size limitations

## Step-by-Step Workflow

### 1. Initial Setup (Already Done)

- ✅ Fixed the interactive test script error
- ✅ Created Hugging Face integration scripts
- ✅ Updated .gitignore to exclude large model files
- ✅ Created documentation

### 2. Push to GitHub

```bash
# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize repository
git init

# Add all files (excluding those in .gitignore)
git add .

# Commit
git commit -m "Initial commit: Bilingual Sentiment Analysis model"

# Create GitHub repository through web interface or GitHub CLI
# Then add remote
git remote add origin https://github.com/YourUsername/BiLingualSentimentMPS.git

# Push
git push -u origin main
```

### 3. Upload Models to Hugging Face

```bash
# Log in to Hugging Face (first time only)
pip install huggingface_hub
huggingface-cli login

# Run the upload script
python upload_to_huggingface.py --username YourUsername
```

### 4. Verify the Workflow

To verify everything works properly:

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/YourUsername/BiLingualSentimentMPS.git
   cd BiLingualSentimentMPS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download models from Hugging Face:
   ```bash
   python download_models_from_hf.py --repo_id YourUsername/bilingual-sentiment-va
   ```

4. Test the models:
   ```bash
   python test_downloaded_models.py --model_dir downloaded_models --model_type distilbert --text "这是一个测试文本，看看情感分析效果如何"
   ```

## Ongoing Development

For future development:

1. **Make code changes**:
   ```bash
   # Make changes to files
   git add .
   git commit -m "Description of changes"
   git push
   ```

2. **Update models**:
   ```bash
   # After retraining models
   python upload_to_huggingface.py --username YourUsername --force
   ```

## Troubleshooting

### Large File Issues

If you accidentally try to commit large files:

```bash
# Remove large files from Git tracking
git rm --cached pretrained_models/**/model.safetensors
git commit -m "Remove large files from tracking"
```

### Hugging Face Authentication

If you encounter authentication issues:

```bash
# Re-login to Hugging Face
huggingface-cli login
# Enter your token when prompted
```

## User Instructions

Add these instructions to your README to help users:

```markdown
## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download models: `python download_models_from_hf.py --repo_id YourUsername/bilingual-sentiment-va`
4. Test the models: `python test_downloaded_models.py --model_type distilbert --text "测试文本"`
```
