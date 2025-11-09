#!/bin/bash

# ============================================================================
# 八卡分布式训练启动脚本
# Distributed Data Parallel Training Script for 8 GPUs
# ============================================================================

# 检查 GPU 数量
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "检测到 GPU 数量: $GPU_COUNT"
echo "Detected GPU count: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "警告: 检测到的 GPU 数量少于 2，建议检查 GPU 配置"
    echo "Warning: Detected GPU count is less than 2, please check GPU configuration"
fi

# 启动分布式训练
# 使用 torchrun，自动根据可用 GPU 数量启动进程
echo "启动 $GPU_COUNT 卡分布式训练..."
echo "Launching distributed training with $GPU_COUNT GPUs..."

torchrun --nproc_per_node=$GPU_COUNT rl_trainer/main_ddp.py "$@"
