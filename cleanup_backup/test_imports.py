import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("Working directory:", os.getcwd())

try:
    from src.models.roberta_model import MultilingualDistilBERTModel, XLMRobertaDistilledModel
    print("Import successful!")
    print("MultilingualDistilBERTModel:", MultilingualDistilBERTModel)
    print("XLMRobertaDistilledModel:", XLMRobertaDistilledModel)
except Exception as e:
    print("Import failed:", e)
    import traceback
    traceback.print_exc()

# Try to inspect the file content
with open("src/models/roberta_model.py", "r", encoding="utf-8") as f:
    content = f.read()
    print("File length:", len(content))
    print("First 100 chars:", repr(content[:100]))
