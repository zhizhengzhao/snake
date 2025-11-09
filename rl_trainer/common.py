import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


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


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Nearest bean relative: (16, 17, 18) - (dx, dy, distance)
# Second nearest bean:   (19, 20, 21) - (dx, dy, distance)
# Snake length:          22:self_length
# Other snake positions: (23, 24) (25, 26) (27, 28) (29, 30) (31, 32) -- relative positions
# Threat scores:        (33, 34) -- threat from opponent snakes
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
    beans_position = np.array(beans_positions, dtype=object)
    beans_position_flat = beans_position.flatten()

    for i in agents_index:
        # [0-1] self head position
        observations[i][:2] = snakes_position[i][0][:]

        # [2-5] head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # [6-15] beans positions (original)
        observations[i][6:16] = beans_position_flat[:]

        # [16-21] 最近的两个豆子的相对位置和距离
        bean_distances = []
        for bean in beans_position:
            dx = bean[1] - head_x
            dy = bean[0] - head_y
            # 考虑环形地图的最短距离
            dx = dx if abs(dx) <= board_width // 2 else dx - np.sign(dx) * board_width
            dy = dy if abs(dy) <= board_height // 2 else dy - np.sign(dy) * board_height
            distance = np.sqrt(dx**2 + dy**2)
            bean_distances.append((dx, dy, distance))

        # 按距离排序，取最近的两个
        bean_distances.sort(key=lambda x: x[2])
        for j in range(min(2, len(bean_distances))):
            observations[i][16 + j*3 : 16 + j*3 + 3] = bean_distances[j]

        # [22] 自身蛇长度（归一化）
        observations[i][22] = len(snakes_position[i]) / 20.0  # 假设最大长度20

        # [23-32] other snake positions (相对位置)
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads_relative = np.delete(snake_heads, i, 0)
        snake_heads_relative_flat = snake_heads_relative.flatten()
        observations[i][23:23+len(snake_heads_relative_flat)] = snake_heads_relative_flat[:]

        # [33-34] threat scores from opponent snakes
        opponent_indices = [j for j in range(len(snakes_position)) if j != i and j >= 3]
        threat_scores = []
        for opp_idx in opponent_indices:
            opp_head = snakes_position[opp_idx][0]
            dx = opp_head[1] - head_x
            dy = opp_head[0] - head_y
            # 考虑环形地图
            dx = dx if abs(dx) <= board_width // 2 else dx - np.sign(dx) * board_width
            dy = dy if abs(dy) <= board_height // 2 else dy - np.sign(dy) * board_height
            distance = np.sqrt(dx**2 + dy**2) + 1e-8
            opp_length = len(snakes_position[opp_idx])
            # 威胁评分 = 1/(distance+1) * (opp_length/self_length)
            threat = (1.0 / (distance + 1)) * (opp_length / max(1, len(snakes_position[i])))
            threat_scores.append(threat)

        # 填充威胁评分（最多两个对手）
        for j, score in enumerate(threat_scores[:2]):
            observations[i][33 + j] = score

    return observations


def get_reward(info, snake_index, reward, score):
    """
    改进的奖励函数（Baseline增强版）

    原始奖励保持：
    - score=1 (胜): +50
    - score=2 (负): -25
    - score=3 (进行中领先): +10
    - score=4 (进行中落后): -5
    - 进食: +20

    新增奖励：
    - 接近豆子奖励: 根据到最近豆子的距离给予小额奖励
    - 存活奖励: 每步存活给予小额奖励
    - 蛇长度奖励: 鼓励蛇长度增长
    """
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))

    for i in snake_index:
        # ===== 原始奖励（保持不变） =====
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 25
        elif score == 3:
            step_reward[i] += 10
        elif score == 4:
            step_reward[i] -= 5

        if reward[i] > 0:
            step_reward[i] += 20
        else:
            # 原始距离惩罚
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            min_dist = min(dists)
            step_reward[i] -= min_dist

            if reward[i] < 0:
                step_reward[i] -= 10

        # ===== 新增奖励设计（不调整现有数值） =====

        # [新增1] 接近豆子的奖励（稍微鼓励向豆子靠近）
        # 只有在没有吃到豆子时才给予，避免与进食奖励冲突
        if reward[i] <= 0:
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            min_dist = min(dists) if dists else 0
            # 反向距离奖励：距离越近，奖励越多（范围 [0, 0.5]）
            proximity_reward = 0.1 * (1.0 / (min_dist + 1.0))
            step_reward[i] += proximity_reward

        # [新增2] 存活奖励（鼓励持续存活）
        # 每步都给予很小的奖励，但如果死亡就不给
        if reward[i] >= 0:
            step_reward[i] += 0.1

        # [新增3] 蛇长度增长奖励（鼓励吃豆子让蛇变长）
        # 注：这个在reward[i] > 0时会被触发
        if reward[i] > 0:
            step_reward[i] += 0.05

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