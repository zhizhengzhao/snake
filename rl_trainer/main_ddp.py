import argparse
import datetime
import os
import sys
import torch
import numpy as np

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from tensorboardX import SummaryWriter
from tqdm import tqdm
from algo.ddpg_ddp import DDPG_DDP
from common import *
from log_path import *
from env.chooseenv import make

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main_worker(local_rank, args):
    """
    每个 GPU 进程执行的主函数

    Args:
        local_rank: 当前进程的本地 rank (0-7 for 8-GPU)
        args: 训练参数
    """
    # 初始化分布式环境
    rank = local_rank  # 在 torchrun 中，local_rank 就是全局 rank
    world_size = torch.cuda.device_count()

    # 为每个进程分配不同的 GPU（必须在 init_process_group 之前）
    torch.cuda.set_device(local_rank)

    # 初始化分布式环境
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

    # 只在 rank 0 进程输出日志
    if rank == 0:
        print("=" * 80)
        print(f"Starting DDP training with {world_size} GPUs")
        print(f"Local rank: {local_rank}, Global rank: {rank}, World size: {world_size}")
        print("=" * 80)
        print(f"\n【训练策略】标准数据并行（Data Parallel）")
        print(f"  - 每卡采样 batch_size: {args.batch_size}")
        print(f"  - 有效 batch_size: {args.batch_size} × {world_size} = {args.batch_size * world_size}")
        print(f"  - 梯度同步: DDP 自动平均（AllReduce / {world_size}）")
        print(f"  - 学习率: {args.a_lr}（保持不变）")
        print(f"  - episodes: {args.max_episodes} (= 单卡50k / {world_size}，总计算量相同）")
        print(f"  - 总梯度计算: {args.max_episodes} × {world_size} ≈ 单卡50k episodes")
        print(f"  - 预期加速: ~{world_size}x（主要受环境采样限制，实际 ~{world_size-1}x）")
        print("=" * 80)

    # 同步所有进程
    torch.distributed.barrier()

    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    if rank == 0:
        print(f'Total agent number: {num_agents}')

    ctrl_agent_index = [0, 1, 2]
    if rank == 0:
        print(f'Agent control by the actor: {ctrl_agent_index}')

    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    if rank == 0:
        print(f'Game board width: {width}')

    height = env.board_height
    if rank == 0:
        print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    if rank == 0:
        print(f'action dimension: {act_dim}')

    obs_dim = 35  # 扩展观测维度
    if rank == 0:
        print(f'observation dimension: {obs_dim}')

    # ================== 第一步：相同的随机种子用于模型初始化 ==================
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 只在 rank 0 进程创建日志路径和 TensorBoard writer
    run_dir = None
    log_dir = None
    writer = None

    if rank == 0:
        run_dir, log_dir = make_logpath(args.game_name, args.algo)
        writer = SummaryWriter(str(log_dir))
        save_config(args, log_dir)
        print(f"Log directory: {log_dir}")

    # 创建模型（使用 DDP 包装）
    # 此时所有进程的模型权重初始化一致（因为用了相同的 seed）
    model = DDPG_DDP(obs_dim, act_dim, ctrl_agent_num, args, local_rank=local_rank)

    # ================== 第二步：不同的随机种子用于数据采样 ==================
    # 这样每个进程会采样不同的 batch，实现真正的方案 B
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if args.load_model:
        if rank == 0:
            print(f"Loading model from run {args.load_model_run}, episode {args.load_model_run_episode}")
        # 计算 load_dir - 所有进程使用相同的路径结构
        base_dir_path = Path(__file__).resolve().parent.parent
        load_dir = os.path.join(str(base_dir_path), 'rl_trainer', 'models', args.game_name,
                               "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    # 同步所有进程
    torch.distributed.barrier()

    episode = 0

    # 进度条（仅在 rank 0 显示）
    pbar = tqdm(total=args.max_episodes, desc="Training", disable=(rank != 0))

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        obs = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        pbar.update(1)
        step = 0
        episode_reward = np.zeros(6)

        while True:

            # ================================== inference ========================================
            logits = model.choose_action(obs)

            # ============================== add opponent actions =================================
            actions = logits_greedy(state_to_training, logits, height, width)

            # Receive reward and observe new state
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)

            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=4)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)

            done = np.array([done] * ctrl_agent_num)

            # ================================== collect data ========================================
            model.replay_buffer.push(obs, logits, step_reward, next_obs, done)

            model.update()

            obs = next_obs
            state_to_training = next_state_to_training
            step += 1

            if args.episode_length <= step or (True in done):

                # 只在 rank 0 进程输出日志和保存模型
                if rank == 0:
                    print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} epsilon: {model.eps:.2f}')
                    print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                          f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')

                    reward_tag = 'reward'
                    loss_tag = 'loss'
                    writer.add_scalars(reward_tag, global_step=episode,
                                       tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})
                    if model.c_loss and model.a_loss:
                        writer.add_scalars(loss_tag, global_step=episode,
                                           tag_scalar_dict={'actor': model.a_loss, 'critic': model.c_loss})

                    if model.c_loss and model.a_loss:
                        print(f'\t\t\t\ta_loss {model.a_loss:.3f} c_loss {model.c_loss:.3f}')

                    if episode % args.save_interval == 0:
                        model.save_model(run_dir, episode)

                env.reset()
                break

        # 同步所有进程（确保所有 GPU 进度一致）
        torch.distributed.barrier()

    # 关闭进度条
    pbar.close()

    # 清理分布式环境
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=6250, type=int,
                       help="DDP版本：默认6250 (= 单卡50000/8)，保持梯度更新数相同")
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    # DDP 相关参数
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed training")

    args = parser.parse_args()

    # torchrun 会自动设置 LOCAL_RANK 环境变量
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.local_rank = local_rank

    main_worker(local_rank, args)
