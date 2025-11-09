# GPU 利用率优化指南

## 问题分析

你观察到增加 batch_size 没有提高 GPU 利用率，这是因为：

### 当前瓶颈

```
当前代码流程（每个 step）：
────────────────────────────
GPU:  |====| (20ms)      |====| (20ms)      GPU 计算
CPU:              ||||||||||| (80ms)                  等待环境
            env.step()                  env.step()

GPU 利用率 = 20 / (20+80) = 20%
即使 batch_size=512，GPU 计算也只增加到 ~30ms，利用率仅提升到 30%
```

**根本原因**：GPU 计算速度非常快（O(batch_size)），而环境采样（游戏引擎）是 CPU 操作，固定耗时 ~80ms。

### 解决方案

**降低更新频率**：不是每 step 都更新，而是积累多个 step 后一次性更新。

```
改进后的流程（update_interval=10）：
────────────────────────────
GPU:                                  |===| (300ms)        GPU 计算（累积10个batch）
CPU:  ||||||| ||||||| ||||||| ... ||||||| 环境采样 x10

GPU 利用率 = 300 / (300+80) = 79%
```

## 使用方法

### 基础命令

```bash
# 默认：每 step 更新一次（原始行为）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py

# 优化：每 10 个 step 更新一次
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --update_interval 10

# 更激进的优化
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --update_interval 20
```

### 推荐配置

```bash
# 平衡配置（推荐）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --batch_size 512 \
  --update_interval 10 \
  --max_episodes 6250

# 更激进的配置（GPU 利用率最高）
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --batch_size 512 \
  --update_interval 20 \
  --max_episodes 6250
```

## 参数选择指南

### update_interval 的影响

| update_interval | GPU 计算时间 | GPU 利用率 | 梯度更新数 | 收敛速度 |
|-----------------|-----------|---------|---------|--------|
| 1（默认） | ~25ms | ~24% | 多（每step） | 快但不稳定 |
| 5 | ~100ms | ~55% | 中 | 平衡 |
| 10（推荐） | ~200ms | ~71% | 少（每10步） | 相对稳定 |
| 20 | ~350ms | ~81% | 很少 | 慢但更稳定 |

### 如何选择？

1. **看 GPU 利用率**：用 `nvidia-smi` 实时监控
   ```bash
   watch -n 1 nvidia-smi
   ```
   - 如果 GPU 利用率 < 40%，增加 `update_interval`
   - 如果已经 > 70%，保持或减小

2. **看收敛速度**：不同 interval 的训练曲线
   - `update_interval=1`：更新频率高，可能噪声大，收敛慢
   - `update_interval=10`：平衡方案，推荐
   - `update_interval=20`：更新频率低，梯度更稳定，但需要更多 episode

3. **看总训练时间**
   ```
   update_interval=1:  每 episode 100 次 update，episode 快速完成但慢
   update_interval=10: 每 episode 10 次 update，episode 慢但 GPU 更优化
   实际总时间：update_interval=10 可能更快
   ```

## 预期性能提升

### 单卡（baseline）
```
batch_size=64, update_interval=1
GPU 利用率：~15%
时间：~100 小时
```

### 8卡（当前配置）
```
batch_size=512, update_interval=1
GPU 利用率：~24%（增加不多）
时间：~14 小时
加速比：~7x
```

### 8卡（优化后）
```
batch_size=512, update_interval=10
GPU 利用率：~70%
时间：~8-10 小时
加速比：~10-12x
```

## 工作原理

### 每 step 更新（update_interval=1）

```python
while True:
    logits = model.choose_action(obs)
    next_state, reward, done, info = env.step(...)
    model.replay_buffer.push(...)
    model.update()  # ← 立即更新，GPU 计算时间短
```

### 降低频率更新（update_interval=10）

```python
update_counter = 0
while True:
    logits = model.choose_action(obs)
    next_state, reward, done, info = env.step(...)
    model.replay_buffer.push(...)

    update_counter += 1
    if update_counter >= 10:
        for _ in range(10):
            model.update()  # ← 一次性做 10 次更新，GPU 计算时间长
        update_counter = 0
```

## 对训练的影响

### 优点
- ✅ GPU 利用率大幅提升（~70% vs ~24%）
- ✅ 总训练时间显著减少（~3 倍加速）
- ✅ 梯度更稳定（多个样本平均）

### 缺点
- ❌ 总梯度更新数减少（100k vs 50k）
- ❌ 需要调整 episode 次数补偿
- ❌ 收敛曲线可能不同

### 建议

如果用 `update_interval=10`，应该相应增加 episodes 来补偿：

```bash
# 原始：50k episodes，每 episode 200 update = 10M 步
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py --max_episodes 6250

# 优化：10k episodes，每 episode 2000 update（10×200），仍然 10M 步
torchrun --nproc_per_node=8 rl_trainer/main_ddp.py \
  --max_episodes 6250 \
  --update_interval 10
  # 梯度更新数 = 6250 × 200 / 10 × 8 ≈ 10M（与原始相同）
```

## 实时监控

```bash
# 新开一个终端，实时查看 GPU 利用率
watch -n 1 nvidia-smi

# 或查看详细 GPU 内存
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -l 1
```

## 故障排除

### 问题：内存溢出（OOM）

**原因**：`update_interval` 太大，累积的数据占用内存太多

**解决**：
```bash
# 减小 update_interval
--update_interval 5

# 或减小 batch_size
--batch_size 256 --update_interval 10
```

### 问题：训练不稳定

**原因**：梯度更新间隔太长，导致梯度分布变化大

**解决**：
```bash
# 减小 update_interval
--update_interval 5

# 或降低学习率
--a_lr 0.00005 --c_lr 0.00005
```

### 问题：GPU 利用率还是低

**原因**：可能不是 GPU 问题，而是环境采样本身的限制

**检查**：
```bash
# 运行时添加详细日志，查看时间分布
# 如果 env.step() 时间 > GPU 计算时间，就无法改进了
```

## 总结

| 配置 | 特点 | 适用场景 |
|-----|------|--------|
| `update_interval=1` | 简单，原始行为 | 单卡训练，不关心 GPU 利用率 |
| `update_interval=5` | 平衡，改进明显 | 8 卡训练，想要快一点 |
| `update_interval=10` | 激进，利用率高 | 8 卡训练，追求速度 |
| `update_interval>20` | 极限优化，可能不稳定 | 仅在 batch_size 很大时使用 |

**推荐**：从 `--update_interval 10` 开始，然后根据 GPU 利用率和训练稳定性调整。
