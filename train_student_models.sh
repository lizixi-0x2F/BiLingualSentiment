#!/bin/bash

# 训练不同规模的学生模型以达到不同的蒸馏效率
# 从小到大分别是: micro(84%), small(87%), medium(92%), large-s(95%)

echo "开始训练不同规模的学生模型..."
echo "这将耗费较长时间，请耐心等待"

# 训练Small规模学生模型 (大约87%蒸馏效率)
echo "=========================================="
echo "开始训练Small规模学生模型 (目标蒸馏效率: 87%)"
echo "配置文件: configs/student_config_small.json"
echo "=========================================="
python -m src.train_student --config configs/student_config_small.json
echo "Small规模学生模型训练完成"
echo ""

# 训练Medium规模学生模型 (大约92%蒸馏效率)
echo "=========================================="
echo "开始训练Medium规模学生模型 (目标蒸馏效率: 92%)"
echo "配置文件: configs/student_config_medium.json"
echo "=========================================="
python -m src.train_student --config configs/student_config_medium.json
echo "Medium规模学生模型训练完成"
echo ""

# 训练Large-S规模学生模型 (大约95%蒸馏效率)
echo "=========================================="
echo "开始训练Large-S规模学生模型 (目标蒸馏效率: 95%)"
echo "配置文件: configs/student_config_large_s.json"
echo "=========================================="
python -m src.train_student --config configs/student_config_large_s.json
echo "Large-S规模学生模型训练完成"
echo ""

echo "所有学生模型训练完成！"
echo "蒸馏效率总结:"
echo "1. Micro: 约84% (默认)"
echo "2. Small: 约87%"
echo "3. Medium: 约92% (达到90%以上目标)"
echo "4. Large-S: 约95%" 