#!/bin/bash
export PYTHONPATH=.
echo "开始LTC-NCP-VA模型训练..."
python src/train.py --config configs/valence_enhanced.yaml
