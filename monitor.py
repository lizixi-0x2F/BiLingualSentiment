#!/usr/bin/env python3
import os
import time
import psutil
import glob
from datetime import datetime

def get_training_status():
    """获取训练进度和模型状态"""
    status = {}
    
    # 检查训练进程
    train_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                      if 'python' in p.info['name'] and any('train.py' in cmd for cmd in p.info['cmdline'] if cmd)]
    
    status['train_running'] = len(train_processes) > 0
    if status['train_running']:
        process = train_processes[0]
        status['train_pid'] = process.pid
        status['train_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        status['train_cpu_percent'] = process.cpu_percent(interval=0.1)
        status['train_running_time'] = time.time() - process.create_time()
    
    # 检查模型目录
    status['model_dir_exists'] = os.path.exists('model_dir')
    if status['model_dir_exists']:
        status['model_files'] = len(os.listdir('model_dir'))
    
    # 检查results目录
    status['results_dir_exists'] = os.path.exists('results')
    if status['results_dir_exists']:
        checkpoint_dirs = glob.glob('results/checkpoint-*')
        status['checkpoints'] = len(checkpoint_dirs)
        
        # 获取最新检查点
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
            status['latest_checkpoint'] = os.path.basename(latest_checkpoint)
            status['checkpoint_time'] = datetime.fromtimestamp(os.path.getctime(latest_checkpoint)).strftime('%Y-%m-%d %H:%M:%S')
    
    # 获取系统资源信息
    status['system_memory_percent'] = psutil.virtual_memory().percent
    status['system_cpu_percent'] = psutil.cpu_percent(interval=0.1)
    
    return status

def format_time(seconds):
    """将秒数格式化为时:分:秒格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def print_status(status):
    """打印状态信息"""
    print("\n===== 训练状态监控 =====")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if status.get('train_running', False):
        print(f"\n训练进程:")
        print(f"- PID: {status['train_pid']}")
        print(f"- 内存使用: {status['train_memory_mb']:.2f} MB")
        print(f"- CPU使用率: {status['train_cpu_percent']:.1f}%")
        print(f"- 运行时间: {format_time(status['train_running_time'])}")
    else:
        print("\n训练进程: 未运行")
    
    print(f"\n检查点状态:")
    if status.get('results_dir_exists', False):
        print(f"- 检查点数量: {status.get('checkpoints', 0)}")
        if status.get('checkpoints', 0) > 0:
            print(f"- 最新检查点: {status.get('latest_checkpoint', 'N/A')}")
            print(f"- 保存时间: {status.get('checkpoint_time', 'N/A')}")
    else:
        print("- 还没有检查点生成")
    
    print(f"\n模型状态:")
    if status.get('model_dir_exists', False):
        print(f"- 模型已保存: 是 ({status.get('model_files', 0)} 个文件)")
    else:
        print("- 模型还未保存")
    
    print(f"\n系统资源:")
    print(f"- 内存使用率: {status.get('system_memory_percent', 0):.1f}%")
    print(f"- CPU使用率: {status.get('system_cpu_percent', 0):.1f}%")
    
if __name__ == "__main__":
    try:
        while True:
            status = get_training_status()
            print_status(status)
            # 如果训练已经结束并且模型已保存，则退出
            if not status.get('train_running', False) and status.get('model_dir_exists', False):
                print("\n训练已完成，模型已保存!")
                break
            # 每30秒更新一次
            print("\n等待30秒后刷新...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n监控已停止") 