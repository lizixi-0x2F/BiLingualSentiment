#!/bin/bash

# 停止所有train.py相关进程的脚本

# 检查是否需要确认
AUTO_CONFIRM=0
if [ "$1" == "-y" ]; then
    AUTO_CONFIRM=1
fi

echo "正在寻找并终止train.py相关进程..."

# 查找所有python train.py进程
PIDS=$(ps aux | grep "python train.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "未找到正在运行的train.py进程"
    exit 0
fi

# 显示进程列表
echo "找到以下train.py进程:"
ps -f $PIDS

# 确认是否终止
if [ $AUTO_CONFIRM -eq 0 ]; then
    read -p "确定要终止这些进程吗? (y/n) " CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        echo "操作已取消"
        exit 0
    fi
else
    echo "自动确认模式，立即终止进程"
fi

# 终止进程
for PID in $PIDS; do
    echo "终止进程 $PID..."
    kill -9 $PID
done

echo "所有train.py进程已终止"
exit 0 