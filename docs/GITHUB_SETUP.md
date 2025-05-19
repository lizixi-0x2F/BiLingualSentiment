# GitHub Setup Instructions

This document outlines how to set up this project on GitHub while handling large model files through Hugging Face.

## 1. Setup with Current Configuration

The project is now configured to:
- Exclude large model files (`.safetensors` and `.bin`) from Git using `.gitignore`
- Keep model configuration files in the repository
- Use Hugging Face for hosting and distributing the trained models

## 2. Push to GitHub

Follow these steps to push the project to GitHub:

```bash
# Initialize Git repository if not already done
git init

# Add all files (excluding those in .gitignore)
git add .

# Commit changes
git commit -m "Initial commit: Bilingual Sentiment Analysis model"

# Add your GitHub repository as remote
git remote add origin https://github.com/YourUsername/BiLingualSentimentMPS.git

# Push to GitHub
git push -u origin main  # or 'master' depending on your default branch name
```

## 3. Upload Models to Hugging Face

After pushing to GitHub, upload the models to Hugging Face:

```bash
# Make sure you're logged in to Hugging Face
huggingface-cli login

# Run the upload script with your Hugging Face username
python upload_to_huggingface.py --username YourUsername
```

## 4. Test the Workflow

To ensure everything works properly:

1. Clone the repository on another machine or clear your local copy
2. Run the download script to get models from Hugging Face:
   ```bash
   python download_models_from_hf.py --repo_id YourUsername/bilingual-sentiment-va
   ```
3. Test the downloaded models:
   ```bash
   python test_downloaded_models.py --text "这是一个测试文本，看看情感分析效果如何"
   ```

## 5. Alternative: Git LFS (Optional)

If you prefer to keep model files in the Git repository, you can use Git Large File Storage (LFS):

1. Install Git LFS: https://git-lfs.github.com/
2. Initialize Git LFS: `git lfs install`
3. Track large files: `git lfs track "*.safetensors" "*.bin"`
4. Modify `.gitignore` to remove the exclusion of these files
5. Commit and push as usual

Note: GitHub has storage limits for LFS, so Hugging Face is still recommended for large model files.
