import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from common import *
from log_path import *
from env.chooseenv import make


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    # ========== 设置GPU设备 ==========
    # 根据命令行参数设置GPU
    from common import get_device
    global device
    device = get_device(gpu_id=args.gpu_id)

    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 29  # 扩展到29维：原26维 + 距离特征 + 长度特征 + 食物数量特征
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = DDPG(obs_dim, act_dim, ctrl_agent_num, args)

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    # ========== 课程学习阶段 ==========
    # 0: 只与贪心蛇对抗
    # 1: 随机混合贪心和随机蛇
    # 2: 更多随机蛇
    curriculum_stage = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:

            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)

            # ============================== add opponent actions (with curriculum learning) =================================
            # 课程学习：随着训练进行，逐步增加对手的难度

            # 更新课程阶段（如果启用课程学习）
            if args.use_curriculum and episode > 0 and episode % args.curriculum_stage_episodes == 0:
                curriculum_stage = min(curriculum_stage + 1, 2)
                if episode % (args.curriculum_stage_episodes * 2) == 0:
                    print(f"[Curriculum Learning] Stage upgraded to: {curriculum_stage}")

            # 根据课程阶段选择对手策略
            if curriculum_stage == 0:
                # 阶段0：只用贪心蛇（相对较弱）
                actions = logits_greedy(state_to_training, logits, height, width)
            elif curriculum_stage == 1:
                # 阶段1：50%概率用贪心蛇，50%用随机蛇（中等难度）
                if np.random.random() < 0.5:
                    actions = logits_greedy(state_to_training, logits, height, width)
                else:
                    actions = logits_random(act_dim, logits)
            else:
                # 阶段2：75%概率用随机蛇，25%用贪心蛇（困难）
                if np.random.random() < 0.75:
                    actions = logits_random(act_dim, logits)
                else:
                    actions = logits_greedy(state_to_training, logits, height, width)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
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
            # Store transition in R
            model.replay_buffer.push(obs, logits, step_reward, next_obs, done)

            model.update()

            obs = next_obs
            state_to_training = next_state_to_training
            step += 1

            if args.episode_length <= step or (True in done):

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ========== 多卡运行配置 ==========
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='使用的GPU ID (0, 1, 2, ...)，-1表示使用CPU')

    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg/dqn")
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    # ========== 改进的超参数 ==========
    parser.add_argument('--buffer_size', default=int(2e5), type=int)  # 增加缓冲区大小（原来1e5）
    parser.add_argument('--tau', default=0.002, type=float)  # 更快的目标网络更新（原来0.001）
    parser.add_argument('--gamma', default=0.99, type=float)  # 更看重未来奖励（原来0.95）
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0002, type=float)  # 更大的学习率加速学习（原来0.0001）
    parser.add_argument('--c_lr', default=0.0002, type=float)  # 更大的学习率加速学习（原来0.0001）
    parser.add_argument('--q_lr', default=0.0002, type=float)  # DQN Q网络学习率
    parser.add_argument('--batch_size', default=128, type=int)  # 更大的批量大小（原来64）
    parser.add_argument('--epsilon', default=0.3, type=float)  # 降低初始探索率（原来0.5）
    parser.add_argument('--epsilon_speed', default=0.99995, type=float)  # 稍快的衰减（原来0.99998）

    # ========== 课程学习参数 ==========
    parser.add_argument('--use_curriculum', action='store_true', help='启用课程学习')
    parser.add_argument('--curriculum_stage_episodes', default=10000, type=int, help='每个课程阶段的episodes数')

    parser.add_argument("--save_interval", default=500, type=int)  # 更频繁地保存（原来1000）
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)
