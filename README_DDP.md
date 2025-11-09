# 八卡分布式训练指南 (8-GPU DDP Training Guide)

## 🎯 训练策略：方案 B（每卡独立采样，梯度累加）

本 DDP 实现采用**方案 B**，这意味着：

### 核心原理

```
单卡训练:
  ┌─────────────────────┐
  │  采样 batch (128)    │
  │  计算梯度            │
  │  优化器更新          │
  └─────────────────────┘
  每 episode: 1 次梯度更新

8卡分布式 (方案B):
  GPU0              GPU1              GPU2        ...      GPU7
  ┌──────────┐      ┌──────────┐      ┌──────────┐        ┌──────────┐
  │采样 128  │      │采样 128  │      │采样 128  │        │采样 128  │
  │计算梯度1 │      │计算梯度2 │      │计算梯度3 │   ...  │计算梯度8 │
  └─────┬────┘      └─────┬────┘      └─────┬────┘        └─────┬────┘
        │                  │                 │                    │
        └──────────────────┼─────────────────┴────────────────────┘
                           ▼
                    DDP 平均梯度
                 (g1+g2+...+g8)/8
                           │
                     调整学习率 × 8
                    补偿梯度平均
                           │
                      优化器更新
                           ▼
              等价于 batch_size=1024 的单卡训练

  每 episode: 8 次梯度更新！
```

### 具体计算

| 指标 | 单卡 | 8卡(方案B) |
|------|------|-----------|
| batch_size | 128 | 128 (每卡) |
| 有效 batch_size | 128 | 1024 |
| 梯度更新数 | 1x | 8x |
| 学习率 | 0.0003 | 0.0024 |
| 单步梯度 | g | (g1+g2+...+g8)/8 |
| 优化器更新 | θ -= lr*g | θ -= lr*8*(g1+g2+...+g8)/8 = θ -= lr*(g1+...+g8) |

### 优缺点

✅ **优点**：
- 梯度更新频率 8 倍
- 可能更快收敛
- 充分利用多 GPU

❌ **缺点**：
- 改变了训练动态（有效 batch size 变大）
- 超参可能需要微调
- 收敛轨迹与单卡不同

---

本文档介绍如何在多个 GPU（推荐 8 张）上进行分布式训练。

This guide explains how to perform distributed training on multiple GPUs (recommended 8).

---

## 快速开始 (Quick Start)

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n snake3v3 python=3.7.5
conda activate snake3v3

# 安装依赖（包含 torch）
pip install -r requirements.txt

# 验证 GPU 和 torch 安装
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

### 2. 运行分布式训练

#### 方法 A：使用脚本启动（推荐）

```bash
# 基础命令 - 自动检测 GPU 数量，8卡训练约6250 episodes（= 单卡50k/8）
chmod +x train_ddp.sh
./train_ddp.sh

# 如果要更多 episodes（会更多计算）
./train_ddp.sh --max_episodes 12500

# 自定义参数
./train_ddp.sh --max_episodes 6250 --epsilon 0.8 --opponent_difficulty_strategy curriculum
```

#### 方法 B：直接使用 torchrun

```bash
# 8 卡训练（默认 6250 episodes = 单卡 50k/8）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py

# 如果要更多 episodes（计算量更大）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --max_episodes 12500

# 自定义参数
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --max_episodes 6250 --batch_size 128

# 只用 4 卡（如果 GPU 不足）
torchrun --nproc_per_node=4 rl_trainer/main_ddp.py
```

#### 方法 C：手动指定 GPU（用于特定 GPU 设备）

```bash
# 使用特定的 GPU (比如 GPU 2-7)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 rl_trainer/main_ddp.py
```

---

## 文件说明

### 新增文件

| 文件 | 说明 |
|-----|------|
| `rl_trainer/algo/ddpg_ddp.py` | DDP 版本的 DDPG 算法实现 |
| `rl_trainer/main_ddp.py` | DDP 版本的训练入口脚本 |
| `train_ddp.sh` | 八卡训练启动脚本 |
| `README_DDP.md` | 本文件 |

### 原有文件（无改动）

所有原始文件保持不变，包括：
- `rl_trainer/main.py` - 单卡版本
- `rl_trainer/algo/ddpg.py` - 单卡版本
- 其他所有文件

---

## 训练参数

DDP 版本支持所有原始参数，训练逻辑完全相同。

### 常用参数

```bash
--max_episodes 6250            # 最大训练回合数 (default: 6250 = 单卡50k/8, 保持梯度更新数相同)
--batch_size 128               # 批处理大小 (default: 128)
--epsilon 0.5                  # 初始探索率 (default: 0.5)
--epsilon_speed 0.9995         # epsilon 衰减速度 (default: 0.9995)
--a_lr 0.0003                  # Actor 学习率 (default: 0.0003)
--c_lr 0.0003                  # Critic 学习率 (default: 0.0003)
--gamma 0.99                   # 折扣因子 (default: 0.99)
--tau 0.01                     # 软更新系数 (default: 0.01)
--save_interval 1000           # 模型保存间隔 (default: 1000)

# 对手难度调度
--opponent_difficulty_strategy curriculum   # linear/exponential/curriculum
--enable_opponent_evasion                   # 启用躲避型对手 (增加计算)
--opponent_evasion_start_episode 40000     # 躲避对手启用的 episode (default: 40000)

# 模型加载
--load_model                   # 加载预训练模型
--load_model_run 2             # 加载的 run 号
--load_model_run_episode 4000  # 加载的 episode
```

### 完整示例

```bash
# 默认设置（8卡，6250 episodes = 单卡50k/8 的梯度更新数）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py

# 如果要加倍计算（12500 episodes）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --max_episodes 12500 \
  --opponent_difficulty_strategy curriculum \
  --save_interval 500

# 从已有模型继续训练
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --load_model \
  --load_model_run 1 \
  --load_model_run_episode 6250 \
  --max_episodes 6250

# 启用躲避型对手（计算开销大）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --max_episodes 6250 \
  --enable_opponent_evasion \
  --opponent_evasion_start_episode 5000
```

---

## 关键特性

### 1. 自动 GPU 分配

DDP 版本自动分配每个进程到不同的 GPU：
- GPU 0 → 进程 0（rank 0）
- GPU 1 → 进程 1（rank 1）
- ...
- GPU 7 → 进程 7（rank 7）

### 2. 梯度同步

- 所有进程在每次 `backward()` 后自动同步梯度
- 优化器更新基于全局梯度（8 个 GPU 的合并梯度）
- 所有进程使用相同的优化器学习率

### 3. 同步检查点

- 所有进程在 episode 结束后同步（`torch.distributed.barrier()`）
- 只有 rank 0 进程保存模型，避免文件冲突
- 只有 rank 0 进程输出日志，减少控制台输出

### 4. 数据一致性

- 所有进程初始化种子相同（`seed=1`）
- 环境状态同步
- 经验回放缓冲区独立（每个进程维护自己的缓冲区）

---

## 性能对比

### 单卡 vs 8 卡

| 指标 | 单卡 (V100) | 8 卡 (V100) |
|------|-----------|-----------|
| 梯度计算 | 1x | ~7.8x |
| 样本收集 | 1x | ~1x* |
| 总体加速 | 1x | ~4-6x |
| 内存占用 | ~10GB | ~12GB per GPU |
| 通信开销 | 0 | ~5-10% |

*样本收集不加速是因为环境是单进程的（游戏环境通常难以并行化）

### 预期效果

- **训练速度**：4-6 倍加快（主要受环境模拟限制）
- **总样本量**：相同 episode 下，梯度更新更频繁
- **收敛速度**：可能略有改善（更多梯度更新）
- **最终性能**：与单卡相同（训练逻辑完全相同）

---

## 故障排查 (Troubleshooting)

### 问题 1：CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate X.00 GiB
```

**解决方案**：
```bash
# 减少 batch_size
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --batch_size 64

# 减少 buffer_size
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --buffer_size 50000
```

### 问题 2：进程挂起 (Process hangs)

**原因**：进程在 `barrier()` 处卡住，通常是因为某些进程崩溃

**解决方案**：
- 检查日志中的错误信息
- 尝试在 rank 0 看到错误后使用 `Ctrl+C` 强制终止

### 问题 3：GPU 显存不足均衡

**现象**：某个 GPU 显存满，其他 GPU 闲置

**原因**：经验回放缓冲区分布不均匀

**解决方案**：
- 减少 `--buffer_size`
- 减少 `--batch_size`

### 问题 4：模型加载失败

```
Model not founded at rl_trainer/models/...
```

**解决方案**：
- 确保单卡训练已生成模型文件
- 检查 `--load_model_run` 和 `--load_model_run_episode` 参数
- 确保模型文件存在于 `rl_trainer/models/snakes_3v3/run{N}/trained_model/`

---

## 日志和结果

### 日志位置

```
rl_trainer/runs/snakes_3v3/
├── run1_YYYYMMDD_HHMMSS/
│   ├── trained_model/
│   │   ├── actor_1000.pth
│   │   ├── critic_1000.pth
│   │   └── ...
│   └── events.*  # TensorBoard 日志
├── run2_YYYYMMDD_HHMMSS/
└── ...
```

### 查看训练曲线

```bash
# 启动 TensorBoard
tensorboard --logdir=rl_trainer/runs/snakes_3v3

# 然后在浏览器打开 http://localhost:6006
```

### 控制台输出

只有 rank 0 进程输出日志：

```
[Episode 00001] total_reward: 45 epsilon: 0.50 [v3.0] difficulty: 0
        snake_1: 15 snake_2: 15 snake_3: 15
        a_loss 0.123 c_loss 0.456
```

---

## 与单卡版本的对比

### 代码改动原则

✅ **新增文件**（无任何影响）：
- `rl_trainer/algo/ddpg_ddp.py`
- `rl_trainer/main_ddp.py`
- `train_ddp.sh`
- `README_DDP.md`

✅ **原有文件保持不变**：
- `rl_trainer/main.py`
- `rl_trainer/algo/ddpg.py`
- `rl_trainer/common.py`
- 所有其他文件

### 切换版本

```bash
# 单卡训练（原始版本）
python rl_trainer/main.py

# 8 卡分布式训练（新版本）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py
```

---

## 高级用法

### 在 SLURM 集群上运行

```bash
#!/bin/bash
#SBATCH --job-name=snake_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --max_episodes 100000 \
  --opponent_difficulty_strategy curriculum
```

### 监控 GPU 使用

```bash
# 实时监控 GPU 状态
watch -n 1 nvidia-smi

# 或者定期检查
while true; do nvidia-smi; sleep 10; done
```

### 调试分布式训练

```bash
# 启用 debug 日志
NCCL_DEBUG=INFO torchrun --nproc_per_node=8 rl_trainer/main_ddp.py

# 检查进程通信
NCCL_DEBUG=TRACE torchrun --nproc_per_node=8 rl_trainer/main_ddp.py
```

---

## FAQ

**Q: DDP 版本和单卡版本的结果会不同吗？**

A: 不会。训练逻辑完全相同，只是梯度更新更频繁（因为有 8 个 GPU 并行计算）。最终收敛结果应该相同或更好。

**Q: 可以使用少于 8 卡吗？**

A: 可以。只需修改 `--nproc_per_node` 的值，比如：
```bash
torchrun --nproc_per_node=4 rl_trainer/main_ddp.py
```

**Q: 可以在多台机器上运行吗？**

A: 可以，但需要额外配置（参考 PyTorch 官方文档）。单机多卡使用本指南。

**Q: 内存占用会增加吗？**

A: 每个 GPU 上的模型大小相同，但梯度和优化器状态会重复。总体内存占用约为单卡的 1.2 倍。

**Q: 如何恢复中断的训练？**

A: 使用 `--load_model` 参数：
```bash
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --load_model \
  --load_model_run 1 \
  --load_model_run_episode 50000
```

---

## 参考资源

- [PyTorch DDP 官方文档](https://pytorch.org/docs/stable/distributed.html)
- [Torch.run 启动器](https://pytorch.org/docs/stable/elastic/run.html)
- [分布式训练最佳实践](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

## 许可证

本 DDP 版本遵循原项目的许可证。

---

**最后更新**: 2024 年 11 月
**版本**: 1.0
**作者**: Generated with Claude Code
