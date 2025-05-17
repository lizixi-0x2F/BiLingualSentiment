#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复混合精度训练中的梯度计算错误
此脚本修改LTC-NCP-RNN模型，解决价值轴反转和tanh应用中的问题
"""

import os
import re

# 模型文件路径
MODEL_FILE = 'ltc_ncp/model.py'

def fix_inplace_operations():
    """修复原地修改操作引起的梯度计算问题"""
    # 读取原始文件
    with open(MODEL_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复模式1: 在应用tanh前检查NaN，避免多次应用tanh
    pattern1 = r"""        # 确保输出在\[-1,1\]范围内
        outputs = torch\.tanh\(outputs\)
        
        # 最终检查输出是否有NaN
        if torch\.isnan\(outputs\)\.any\(\):
            print\("警告: 最终输出包含NaN值，使用0替换"\) if DEBUG else None
            outputs = torch\.nan_to_num\(outputs, nan=0\.0\)
            # 应用tanh确保在\[-1,1\]范围内
            outputs = torch\.tanh\(outputs\)"""
    
    replacement1 = """        # 处理NaN值(如果存在)
        if torch.isnan(outputs).any():
            print("警告: 输出包含NaN值，使用0替换") if DEBUG else None
            outputs = torch.nan_to_num(outputs, nan=0.0)
            
        # 确保输出在[-1,1]范围内 - 仅应用一次tanh
        outputs = torch.tanh(outputs)"""
    
    # 修复模式2: 避免价值轴反转中的原地修改
    pattern2 = r"""        # 应用价值和效度反转（如果启用）
        if self\.invert_valence:
            outputs\[:, 0\] = -outputs\[:, 0\]  # 反转价值维度
        
        if self\.invert_arousal:
            outputs\[:, 1\] = -outputs\[:, 1\]  # 反转效度维度"""
    
    replacement2 = """        # 应用价值和效度反转（如果启用）- 避免原地修改
        if self.invert_valence or self.invert_arousal:
            # 创建新的张量，避免原地修改
            outputs_clone = outputs.clone()
            
            if self.invert_valence:
                outputs_clone[:, 0] = -outputs[:, 0]  # 反转价值维度
            
            if self.invert_arousal:
                outputs_clone[:, 1] = -outputs[:, 1]  # 反转效度维度
                
            outputs = outputs_clone"""
    
    # 应用修复
    new_content = re.sub(pattern1, replacement1, content)
    new_content = re.sub(pattern2, replacement2, new_content)
    
    # 如果有修改，保存文件
    if new_content != content:
        # 先创建备份
        backup_file = MODEL_FILE + '.bak'
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已创建原始文件备份: {backup_file}")
        
        # 保存修改后的文件
        with open(MODEL_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"成功修复 {MODEL_FILE} 中的原地操作问题")
        return True
    else:
        print("没有找到需要修复的模式，或模式已经修复")
        return False

if __name__ == "__main__":
    print("开始修复混合精度训练中的梯度计算错误...")
    success = fix_inplace_operations()
    if success:
        print("修复完成！请重新运行训练脚本。")
    else:
        print("无需修复或修复失败，请手动检查模型代码。") 