#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix indentation in the LinformerMiniTransformer file.
"""

import re

def fix_file_indentation(filename):
    """Fix indentation in a file by standardizing all indentation to 4 spaces."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the LinformerMiniTransformer class definition
    class_match = re.search(r'class LinformerMiniTransformer\(nn\.Module\):', content)
    if not class_match:
        print("Could not find LinformerMiniTransformer class definition!")
        return False
    
    # Find the forward method definition outside the class
    forward_match = re.search(r'def forward\(self, src, src_mask=None, src_key_padding_mask=None\):', content)
    if not forward_match:
        print("Could not find forward method definition!")
        return False
    
    # Fix file content - ensure proper indentation
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_class = False
    class_indent = ""
    method_indent = ""
    fixed_lines = []
    
    for line in lines:
        # Check for class definition
        if "class LinformerMiniTransformer(nn.Module):" in line:
            in_class = True
            class_indent = line.split("class")[0]
            method_indent = class_indent + "    "  # 4 spaces standard indentation
        
        # Check for forward method outside class
        if not line.startswith(method_indent) and "def forward(self, src, src_mask=None, src_key_padding_mask=None):" in line:
            # This is the forward method defined outside the class - fix its indentation
            line = method_indent + line.lstrip()
        
        # If we're inside a method in the class, ensure consistent indentation
        if in_class and line.strip() and not line.startswith(class_indent) and not line.startswith(method_indent):
            # This line might have improper indentation - calculate proper level
            stripped = line.lstrip()
            leading_spaces = len(line) - len(stripped)
            
            if leading_spaces == 0:
                if stripped.startswith("def "):
                    # This is a method definition, use method_indent
                    line = method_indent + stripped
                else:
                    # We're back at class level
                    line = class_indent + stripped
            elif leading_spaces > 0:
                # This is indented content inside a method
                if stripped.startswith("def "):
                    # This is a nested method definition (unlikely but possible)
                    line = method_indent + stripped
                else:
                    # Regular content inside a method - add additional level
                    line = method_indent + "    " + stripped
        
        fixed_lines.append(line)
    
    # Write fixed content back to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {filename}")
    return True

if __name__ == "__main__":
    file_path = r"c:\Users\User\Documents\GitHubRes\BiLingualSentimentMPS\src\core\linformer_mini_transformer.py"
    fixed = fix_file_indentation(file_path)
    if fixed:
        print("Successfully fixed indentation issues.")
    else:
        print("Failed to fix indentation issues.")
