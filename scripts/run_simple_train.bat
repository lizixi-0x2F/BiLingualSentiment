@echo off
REM LTC-NCP-VA 简化版训练脚本 (Windows Batch版)
REM 移除了高级功能：FGM对抗训练、软标签、多任务学习头

echo =====================================================
echo   LTC-NCP-VA 简化版训练启动脚本
echo =====================================================
echo 模型: 基础版情感价效度分析模型
echo 配置: configs/simple_base.yaml
echo 特性: 基础版(无高级功能)
echo -----------------------------------------------------

REM 设置运行参数
set EPOCHS=30
set BATCH_SIZE=32
set LEARNING_RATE=0.001
set OUTPUT_DIR=results/simple_base

REM 创建目录
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM 记录开始时间
echo 开始时间: %date% %time% >> %OUTPUT_DIR%/training_log.txt
echo 开始时间: %date% %time%

REM 运行训练脚本
echo 正在启动训练...
python src/train_simple.py ^
  --config configs/simple_base.yaml ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --lr %LEARNING_RATE% ^
  --save_dir %OUTPUT_DIR%

REM 记录结束时间
echo 结束时间: %date% %time% >> %OUTPUT_DIR%/training_log.txt
echo 结束时间: %date% %time%

echo =====================================================
echo   训练完成！结果保存在: %OUTPUT_DIR%
echo =====================================================

pause
