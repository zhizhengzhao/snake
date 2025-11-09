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
from common import get_opponent_difficulty

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
        print(f"\n【训练策略】方案 B：每卡独立采样，梯度累加")
        print(f"  - batch_size: {args.batch_size} × {world_size} GPUs = 有效 batch_size {args.batch_size * world_size}")
        print(f"  - 学习率: {args.a_lr} × {world_size} = {args.a_lr * world_size}")
        print(f"  - 梯度更新频率: {world_size}x（每 episode 有 {world_size} 倍的梯度更新）")
        print(f"  - 预期效果: 更新频率高，可能更快收敛，但超参需验证")
        print("=" * 80)
        print("==algo: ", args.algo)
        print(f'device: cuda:{local_rank}')
        print(f'model episode: {args.model_episode}')
        print(f'save interval: {args.save_interval}')
        print(f'[v3.0] opponent difficulty strategy: {args.opponent_difficulty_strategy}')

    # 同步所有进程
    torch.distributed.barrier()

    # 创建环境（所有进程都创建同一个环境，但只有 rank 0 保存日志）
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

    obs_dim = 30
    if rank == 0:
        print(f'observation dimension: {obs_dim}')

    # 所有进程使用相同的随机种子以确保初始化一致
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
    model = DDPG_DDP(obs_dim, act_dim, ctrl_agent_num, args, local_rank=local_rank)

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

        # [v3.0-v3.1] 计算本轮难度
        opponent_difficulty = get_opponent_difficulty(episode, args.max_episodes, args.opponent_difficulty_strategy)
        enable_evasion = args.enable_opponent_evasion and episode >= args.opponent_evasion_start_episode

        while True:

            # ================================== inference ========================================
            logits = model.choose_action(obs)

            # ============================== add opponent actions =================================
            actions = logits_greedy(state_to_training, logits, height, width,
                                   opponent_difficulty=opponent_difficulty, enable_evasion=enable_evasion)

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
                    evasion_str = ' [v3.1] evasion:ON' if enable_evasion else ''
                    print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} epsilon: {model.eps:.2f} [v3.0] difficulty: {opponent_difficulty}{evasion_str}')
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
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.01, type=float, help="soft update coefficient (default: 0.01, was 0.001)")
    parser.add_argument('--gamma', default=0.99, type=float, help="discount factor (default: 0.99, was 0.95)")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0003, type=float, help="actor learning rate (default: 0.0003, was 0.0001)")
    parser.add_argument('--c_lr', default=0.0003, type=float, help="critic learning rate (default: 0.0003, was 0.0001)")
    parser.add_argument('--batch_size', default=128, type=int, help="batch size (default: 128, was 64)")
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.9995, type=float, help="epsilon decay (default: 0.9995, was 0.99998)")

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    # [v3.0-v3.1] 对手难度参数
    parser.add_argument('--opponent_difficulty_strategy', default='curriculum', type=str,
                       help="difficulty schedule strategy: linear/exponential/curriculum")
    parser.add_argument('--enable_opponent_evasion', action='store_true',
                       help="[v3.1] enable evasion-aware opponent (path planning). WARNING: significant computation overhead!")
    parser.add_argument('--opponent_evasion_start_episode', default=40000, type=int,
                       help="episode to start using evasion opponent (default: 40k, disabled if --enable_opponent_evasion not set)")

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
