import sys
import os
sys.path.insert(0, os.getcwd())
print('Current working directory:', os.getcwd())

try:
    print('Trying to import modules...')
    import src.models.roberta_model
    print('Module imported successfully')
    print('Module contains:', dir(src.models.roberta_model))
except Exception as e:
    print('Error importing module:', e)
    import traceback
    traceback.print_exc()
