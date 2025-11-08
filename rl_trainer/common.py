import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml

# ========== 多卡支持 ==========
# 自动选择可用的GPU，如果没有GPU则使用CPU
def get_device(gpu_id=None):
    """
    获取设备配置

    参数:
        gpu_id: 指定GPU ID (0, 1, 2...)
                如果为None，自动选择第一个可用的GPU
                如果为-1，强制使用CPU

    返回:
        torch.device对象
    """
    if gpu_id == -1:
        # 强制使用CPU
        return torch.device("cpu")

    if not torch.cuda.is_available():
        # 没有GPU，使用CPU
        print("⚠️ CUDA不可用，使用CPU")
        return torch.device("cpu")

    if gpu_id is None:
        # 自动选择第一个可用GPU
        gpu_id = 0

    # 检查指定的GPU是否可用
    if gpu_id >= torch.cuda.device_count():
        print(f"⚠️ GPU {gpu_id} 不可用，自动切换到GPU 0")
        gpu_id = 0

    device = torch.device(f"cuda:{gpu_id}")
    print(f"✅ 使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    return device

# 默认使用第一个GPU或CPU
device = get_device(gpu_id=0)


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


# ========== 改进的观察特征 ==========
# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
#
# 新增特征：
# [26]: 到最近食物的距离（归一化）
# [27]: 蛇的长度（身体段数）
# [28]: 游戏中剩余的食物数量（归一化）
def get_observations(state, agents_index, obs_dim, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()

    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:26] = snake_heads.flatten()[:]

        # ========== 新增特征：距离、长度、食物数量 ==========
        if obs_dim > 26:
            # 到最近食物的距离（归一化到0-1之间）
            self_head = np.array(snakes_position[i][0])
            if len(beans_position) > 0:
                dists = [np.sqrt(np.sum(np.square(bean_pos - self_head))) for bean_pos in beans_positions]
                min_dist = min(dists) if dists else 0
                # 距离归一化：最大可能距离约为sqrt(20^2 + 10^2) ≈ 22.36
                observations[i][26] = min_dist / 25.0
            else:
                observations[i][26] = 0

            # 蛇的长度（当前长度 / 初始长度，归一化）
            # 初始长度为3，最大长度约为50
            snake_length = len(snakes_position[i])
            observations[i][27] = min(snake_length / 50.0, 1.0)

            # 食物数量（当前数量 / 最大数量）
            observations[i][28] = len(beans_positions) / 5.0  # 最多5个食物

    return observations


def get_reward(info, snake_index, reward, score):
    """
    超密集奖励函数设计 - 最大化学习信号密度

    设计原理：
    1. 将长期奖励分解为多个阶段奖励
    2. 添加细粒度的距离阶段奖励
    3. 添加增长激励（蛇变长有奖励）
    4. 添加协作奖励（队友进食有奖励）
    5. 添加安全距离奖励（远离对手）
    6. 添加食物关注度奖励（主动追踪食物）
    7. 添加能量衰减惩罚（鼓励快速进食）

    期望效果：
    - 每一步都有多个信号的叠加
    - 更细致的反馈，更快的收敛
    - 更强的目标导向性
    """
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))

    for i in snake_index:
        # ===== 1. 游戏结束时的结果奖励（最高优先级）=====
        if score == 1:
            step_reward[i] += 200  # 赢了！（↑ 从150提升到200）
        elif score == 2:
            step_reward[i] -= 80   # 输了（↑ 从-50降到-80，更强的惩罚）
        elif score == 3:
            step_reward[i] += 40   # 正在赢（↑ 从20提升到40）
        elif score == 4:
            step_reward[i] -= 15   # 正在输（↑ 从-8降到-15）

        # ===== 2. 吃食物的奖励（核心激励）=====
        if reward[i] > 0:
            step_reward[i] += 80  # 吃到食物！（↑ 从60提升到100）

            # 2.1 额外的成就奖励 - 连续进食奖励
            # 获取当前蛇的长度作为代理变量
            current_snake_length = len(snakes_position[i])
            if current_snake_length >= 5:
                step_reward[i] += 20  # 蛇长度≥5，额外奖励
            if current_snake_length >= 8:
                step_reward[i] += 30  # 蛇长度≥8，额外奖励

        else:
            # ===== 3. 距离奖励系统（最重要的导向信号）=====
            self_head = np.array(snake_heads[i])

            # 3.1 与食物的距离奖励
            if len(beans_position) > 0:
                dists = [np.sqrt(np.sum(np.square(other_head - self_head)))
                         for other_head in beans_position]
                min_dist = min(dists) if dists else 1.0

                # 分阶段的距离奖励 - 更密集的反馈
                if min_dist <= 2:
                    step_reward[i] += 8    # 非常接近食物：+8
                elif min_dist <= 4:
                    step_reward[i] += 5    # 接近食物：+5
                elif min_dist <= 6:
                    step_reward[i] += 3    # 中等距离：+3
                elif min_dist <= 10:
                    step_reward[i] += 1    # 较远距离：+1
                else:
                    step_reward[i] -= 0.5  # 很远距离：-0.5（轻微惩罚）

                # 3.2 速度奖励 - 激励快速靠近食物
                # 通过蛇长度代理：长的蛇应该优先进食
                snake_length = len(snakes_position[i])
                if min_dist <= 3 and snake_length >= 5:
                    step_reward[i] += 5  # 长蛇靠近食物额外奖励

            # ===== 4. 碰撞系统 =====
            if reward[i] < 0:
                step_reward[i] -= 8  # 碰撞惩罚（↑ 从-5提升到-8）

            # 4.1 安全距离奖励 - 远离对手
            other_snake_heads = np.delete(snake_heads, i, 0) if len(snake_heads) > 1 else []
            if len(other_snake_heads) > 0:
                min_dist_to_opponent = min([np.sqrt(np.sum(np.square(other_head - self_head)))
                                           for other_head in other_snake_heads])
                if min_dist_to_opponent >= 5:
                    step_reward[i] += 1  # 远离对手有奖励
                elif min_dist_to_opponent <= 2:
                    step_reward[i] -= 3  # 靠近对手有惩罚

        # ===== 5. 生存与活动奖励（持续激励）=====
        step_reward[i] += 0.2  # 每活一步基础奖励（↑ 从0.1提升到0.2）

        # 5.1 长蛇奖励 - 激励蛇长大
        current_length = len(snakes_position[i])
        if current_length > 3:  # 长度超过初始值
            length_bonus = (current_length - 3) * 0.5  # 每多一段身体+0.5
            step_reward[i] += min(length_bonus, 5)  # 上限为5

        # 5.2 存活步数奖励 - 激励长期生存
        # 这在回合内部会自动累加
        if score == 0:  # 游戏未结束
            step_reward[i] += 0.05  # 额外生存奖励

        # ===== 6. 食物关注奖励 - 激励主动追踪 =====
        # 如果当前有多个食物，奖励关注最近的食物
        self_head_for_food = np.array(snake_heads[i])
        if len(beans_position) > 0:
            dists = [np.sqrt(np.sum(np.square(other_head - self_head_for_food)))
                     for other_head in beans_position]
            min_dist = min(dists)

            # 当距离减小时给奖励（表示正在靠近食物）
            # 这需要对比前一步，这里用一个简化的版本
            if min_dist < 5:
                step_reward[i] += 0.3  # 在食物感知范围内有额外奖励

        # ===== 7. 队友进食奖励（可选，激励团队合作）=====
        # 获取同队伍的其他蛇（snake_index通常包含同队蛇）
        team_reward_bonus = 0
        for j in snake_index:
            if i != j and reward[j] > 0:
                team_reward_bonus += 5  # 队友进食也有奖励（团队协作）
        step_reward[i] += team_reward_bonus

    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions


def logits_greedy(state, logits, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()