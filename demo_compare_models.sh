#!/bin/bash

# 定义测试文本
TEXT=${1:-"我今天非常开心，因为我完成了所有工作。"}
echo "测试文本: \"$TEXT\""
echo ""

# 创建临时目录(如果不存在)
mkdir -p checkpoints/student_small
mkdir -p checkpoints/student_medium
mkdir -p checkpoints/student_large_s

# 创建临时符号链接(如果文件不存在)
[ ! -f checkpoints/student_small/best_model.pt ] && ln -s ../student/best_model.pt checkpoints/student_small/best_model.pt
[ ! -f checkpoints/student_medium/best_model.pt ] && ln -s ../student/best_model.pt checkpoints/student_medium/best_model.pt
[ ! -f checkpoints/student_large_s/best_model.pt ] && ln -s ../student/best_model.pt checkpoints/student_large_s/best_model.pt

# 对比各种规模模型效果
echo "=================== 教师模型 (100% 效率) ==================="
./run_inference.sh teacher "" "$TEXT"
echo ""

echo "================= Micro 学生模型 (84% 效率) ================"
./run_inference.sh student micro "$TEXT"
echo ""

echo "================ Small 学生模型 (87% 效率) ================="
./run_inference.sh student small "$TEXT"
echo ""

echo "================ Medium 学生模型 (92% 效率) ================"
./run_inference.sh student medium "$TEXT"
echo ""

echo "================ Large-S 学生模型 (95% 效率) =============="
./run_inference.sh student large "$TEXT"
echo ""

echo "模型比较完成！"
echo "注意：目前这些结果使用的是相同的模型权重，"
echo "要获得真正的效率差异，需完成训练不同规模的学生模型。"
echo "可以使用 ./train_student_models.sh 训练不同规模的模型。"