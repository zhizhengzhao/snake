import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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
# Nearest bean relative: [6, 7, 8] (dx, dy, distance)
# Next nearest bean relative: [9, 10, 11] (dx, dy, distance)
# Threat assessment: [12, 13] (enemy1_threat, enemy2_threat)
# Original beans: [14, 25]
# Original snakes: [26, 29]
def get_observations(state, agents_index, obs_dim, height, width):
    """
    扩展观测空间 - 26维 → 30维，包含相对距离特征和威胁评分

    新增特征：
    - [6-8]: 最近豆子的相对位置 (dx, dy, distance)
    - [9-11]: 次近豆子的相对位置 (dx, dy, distance)
    - [12-13]: 两个对手蛇的威胁评分
    """
    def _get_nearest_beans(head_x, head_y, beans_positions, board_width, board_height, top_k=2):
        """获取最近的K个豆子的相对位置信息"""
        if len(beans_positions) == 0:
            return [(0, 0, board_width * 2) for _ in range(top_k)]

        bean_info = []
        for bean_y, bean_x in beans_positions:
            dx = bean_x - head_x
            dy = bean_y - head_y
            # 考虑环形地图的最短距离
            dx = dx if abs(dx) <= board_width // 2 else dx - np.sign(dx) * board_width
            dy = dy if abs(dy) <= board_height // 2 else dy - np.sign(dy) * board_height
            distance = np.sqrt(dx**2 + dy**2)
            bean_info.append((dx, dy, distance))

        # 按距离排序，取最近的K个
        bean_info.sort(key=lambda x: x[2])
        nearest = bean_info[:top_k]

        # 补充缺失的豆子信息
        while len(nearest) < top_k:
            nearest.append((0, 0, board_width * 2))

        return nearest

    def _get_threat_info(head_pos, snakes_positions, agent_index, board_width, board_height):
        """评估来自其他蛇的威胁程度"""
        threat_scores = []
        self_head = np.array(head_pos)

        for i, snake_pos in enumerate(snakes_positions):
            if i == agent_index:
                continue

            enemy_head = np.array(snake_pos[0])
            dx = enemy_head[1] - self_head[1]
            dy = enemy_head[0] - self_head[0]

            # 考虑环形地图的最短距离
            dx = dx if abs(dx) <= board_width // 2 else dx - np.sign(dx) * board_width
            dy = dy if abs(dy) <= board_height // 2 else dy - np.sign(dy) * board_height

            distance = np.sqrt(dx**2 + dy**2) + 1e-8
            snake_length = len(snake_pos)

            # 威胁评分：(距离倒数) * (蛇长度因子)
            threat_score = (1.0 / (distance + 1)) * (snake_length / 3.0)
            threat_scores.append(threat_score)

        # 补充缺失的威胁信息
        while len(threat_scores) < 2:
            threat_scores.append(0.0)

        return threat_scores[:2]

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

    for i in agents_index:
        # [0-1] self head position
        observations[i][:2] = snakes_position[i][0][:]

        # [2-5] head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # [6-11] 最近和次近豆子的相对位置
        nearest_beans = _get_nearest_beans(head_x, head_y, beans_position, board_width, board_height, top_k=2)
        for j, (dx, dy, dist) in enumerate(nearest_beans):
            base_idx = 6 + j * 3
            observations[i][base_idx] = dx
            observations[i][base_idx + 1] = dy
            observations[i][base_idx + 2] = dist

        # [12-13] 威胁评分
        threat_scores = _get_threat_info(snakes_position[i][0], snakes_position, i, board_width, board_height)
        observations[i][12:14] = threat_scores

        # [14-25] 原始豆子位置 (兼容性)
        beans_flat = beans_position.flatten()
        observations[i][14:min(14 + len(beans_flat), 26)] = beans_flat[:min(len(beans_flat), 12)]

        # [26-29] 对手蛇头相对位置
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads_relative = np.delete(snake_heads, i, 0)
        snake_heads_flat = snake_heads_relative.flatten()[:4]
        observations[i][26:26 + len(snake_heads_flat)] = snake_heads_flat

    return observations


# def get_reward(info, snake_index, reward, score):
#     snakes_position = np.array(info['snakes_position'], dtype=object)
#     beans_position = np.array(info['beans_position'], dtype=object)
#     snake_heads = [snake[0] for snake in snakes_position]
#     step_reward = np.zeros(len(snake_index))
#     for i in snake_index:
#         if score == 1:
#             step_reward[i] += 50
#         elif score == 2:
#             step_reward[i] -= 25
#         elif score == 3:
#             step_reward[i] += 10
#         elif score == 4:
#             step_reward[i] -= 5

#         if reward[i] > 0:
#             step_reward[i] += 20
#         else:
#             self_head = np.array(snake_heads[i])
#             dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
#             step_reward[i] -= min(dists)
#             if reward[i] < 0:
#                 step_reward[i] -= 10

#     return step_reward


def _greedy_level1(state, beans, snakes, width, height):
    """
    [v3.0] Level 1 低级贪心 - 追豆但不避碰撞
    特点：只计算距离，忽视碰撞危险
    """
    actions = []
    for i in [3, 4, 5]:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        bean_x, bean_y, _ = get_min_bean(head_x, head_y, beans)

        # 计算四个方向到豆子的距离 (不检查碰撞)
        next_distances = []
        next_distances.append(math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2))
        next_distances.append(math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2))
        next_distances.append(math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2))
        next_distances.append(math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2))
        actions.append(next_distances.index(min(next_distances)))

    return actions


def _greedy_level3(state, beans, snakes, width, height):
    """
    [v3.0] Level 3 高级贪心 - 加入风险评估系数
    特点：追豆同时评估周围威胁程度，优先躲避被包围
    """
    actions = []
    for i in [3, 4, 5]:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        bean_x, bean_y, _ = get_min_bean(head_x, head_y, beans)

        next_distances = []
        next_risk_penalties = []

        # 评估四个方向
        directions = [
            ((head_x, (head_y - 1) % height), 0),  # up
            ((head_x, (head_y + 1) % height), 1),  # down
            (((head_x - 1) % width, head_y), 2),   # left
            (((head_x + 1) % width, head_y), 3),   # right
        ]

        for (nx, ny), dir_idx in directions:
            # 距离评分
            dist = math.sqrt((nx - bean_x) ** 2 + (ny - bean_y) ** 2)
            next_distances.append(dist)

            # 风险评分：被包围程度 + 是否即将碰撞
            surrounding_value = head_surrounding[dir_idx]
            collision_risk = 0 if surrounding_value <= 1 else 10  # 碰撞直接惩罚

            # 计算周围威胁程度
            nearby_threats = sum(1 for s_val in head_surrounding if s_val > 1) * 2

            total_risk = collision_risk + nearby_threats
            next_risk_penalties.append(total_risk)

        # 联合评分 = 距离 + 风险权重
        combined_scores = [d + 0.5 * r for d, r in zip(next_distances, next_risk_penalties)]
        actions.append(combined_scores.index(min(combined_scores)))

    return actions


def _greedy_level4(state, beans, snakes, width, height):
    """
    [v3.0] Level 4 精英贪心 - 多目标评估 + 动态切换
    特点：同时评估豆子、威胁、己蛇长度，动态在侵略和保守间切换
    """
    actions = []
    for i in [3, 4, 5]:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        bean_x, bean_y, _ = get_min_bean(head_x, head_y, beans)

        # 计算自蛇长度因子 (长蛇更保守)
        snake_length = len(snakes[i])
        aggression = max(0.3, 1.0 - (snake_length / 20.0))  # 长蛇更保守

        next_distances = []
        next_scores = []

        directions = [
            ((head_x, (head_y - 1) % height), 0),  # up
            ((head_x, (head_y + 1) % height), 1),  # down
            (((head_x - 1) % width, head_y), 2),   # left
            (((head_x + 1) % width, head_y), 3),   # right
        ]

        for (nx, ny), dir_idx in directions:
            # 距离评分 (归一化)
            dist = math.sqrt((nx - bean_x) ** 2 + (ny - bean_y) ** 2)
            next_distances.append(dist)

            # 检查碰撞和包围风险
            surrounding_value = head_surrounding[dir_idx]
            if surrounding_value > 1:
                # 即将碰撞，直接排除 (赋予最高分)
                next_scores.append(1000.0)
            else:
                # 没有即将碰撞的危险
                # 评估周围被包围程度
                blocked_count = sum(1 for s_val in head_surrounding if s_val > 1)
                encirclement_penalty = blocked_count * 5

                # 联合评分 (距离主导，威胁调节)
                score = (dist * aggression) + (encirclement_penalty * (1 - aggression))
                next_scores.append(score)

        actions.append(next_scores.index(min(next_scores)))

    return actions


def _get_safe_paths(state, snake_head, width, height, max_steps=3):
    """
    [v3.1] 计算安全路径集合 - 使用BFS找出N步内不会碰撞的所有路径

    返回：可行动作列表 (0-3) 和对应的安全评分
    """
    from collections import deque

    x, y = snake_head[1], snake_head[0]
    safe_actions = []

    # BFS遍历，找出所有安全路径
    queue = deque([(x, y, 0, [])])  # (x, y, steps, action_path)
    visited = set()
    visited.add((x, y, 0))

    while queue:
        cx, cy, steps, path = queue.popleft()

        if steps >= max_steps:
            if len(path) > 0:
                safe_actions.append(path[0])
            continue

        # 检查四个方向
        directions = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]  # up, down, left, right

        for dx, dy, action_idx in directions:
            nx = (cx + dx) % width
            ny = (cy + dy) % height

            # 检查是否碰撞
            cell_value = state[ny][nx]
            if cell_value <= 1:  # 0=空或1=豆子
                state_key = (nx, ny, steps + 1)
                if state_key not in visited:
                    visited.add(state_key)
                    new_path = path + [action_idx]
                    queue.append((nx, ny, steps + 1, new_path))

    return safe_actions if safe_actions else [0, 1, 2, 3]  # 默认所有动作都可用


def _greedy_level_evasion(state, beans, snakes, width, height):
    """
    [v3.1] 躲避型对手 - 贪心追豆 + 路径规划式躲避

    特点：
    - 优先选择追豆且安全的路径
    - 使用BFS找出安全移动空间
    - 在多个安全选择中选择最靠近豆子的
    """
    actions = []

    for i in [3, 4, 5]:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        bean_x, bean_y, _ = get_min_bean(head_x, head_y, beans)

        # 获取安全动作集合
        safe_actions = _get_safe_paths(state, (head_y, head_x), width, height, max_steps=2)

        if not safe_actions:
            # 如果没有安全动作，随机选择
            actions.append(np.random.randint(4))
            continue

        # 在安全动作中选择最靠近豆子的
        best_action = safe_actions[0]
        best_distance = math.inf

        for action in safe_actions:
            # 计算该动作后的新位置到豆子的距离
            if action == 0:  # up
                new_x, new_y = head_x, (head_y - 1) % height
            elif action == 1:  # down
                new_x, new_y = head_x, (head_y + 1) % height
            elif action == 2:  # left
                new_x, new_y = (head_x - 1) % width, head_y
            else:  # right
                new_x, new_y = (head_x + 1) % width, head_y

            dist = math.sqrt((new_x - bean_x) ** 2 + (new_y - bean_y) ** 2)
            if dist < best_distance:
                best_distance = dist
                best_action = action

        actions.append(best_action)

    return actions


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions


def logits_greedy(state, logits, height, width, opponent_difficulty=0, enable_evasion=False):
    """
    [v3.0-v3.1] 动态对手难度梯度 + 躲避型对手

    难度级别 (0-4)：
    - Level 0: 纯随机策略 (随机动作)
    - Level 1: 低级贪心 (目标固定豆子，不避碰撞)
    - Level 2: 中级贪心 (标准贪心-追豆+简单躲避)
    - Level 3: 高级贪心 (优化贪心+评估风险)
    - Level 4: 精英贪心 (多目标评估+动态切换)

    参数：
    - enable_evasion: 启用[v3.1]躲避型对手 (路径规划式躲避，比Level 4更强)

    使用方式：
    - 训练早期：opponent_difficulty=0 (易对手，快速学习)
    - 训练中期：opponent_difficulty=2 (平衡对手)
    - 训练后期：opponent_difficulty=4, enable_evasion=True (强对手，精锐学习)
    """
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
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])

    # 根据难度选择对手策略
    if enable_evasion:
        # [v3.1] 躲避型对手 (启用路径规划躲避)
        opponent_action = _greedy_level_evasion(state, beans, snakes, width, height)
    elif opponent_difficulty == 0:
        # Level 0: 纯随机
        opponent_action = np.random.randint(4, size=3)
    elif opponent_difficulty == 1:
        # Level 1: 低级贪心 (追豆，不避碰撞)
        opponent_action = _greedy_level1(state, beans, snakes, width, height)
    elif opponent_difficulty == 2:
        # Level 2: 标准贪心 (本体)
        opponent_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])
    elif opponent_difficulty == 3:
        # Level 3: 高级贪心 (优化风险评估)
        opponent_action = _greedy_level3(state, beans, snakes, width, height)
    else:  # opponent_difficulty >= 4
        # Level 4: 精英贪心 (多目标评估)
        opponent_action = _greedy_level4(state, beans, snakes, width, height)

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = opponent_action

    return action_list


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def get_opponent_difficulty(episode, max_episodes, difficulty_strategy='linear'):
    """
    [v3.0] 计算动态对手难度等级

    难度调度策略：
    - 'linear': 线性递进 (0→1→2→3→4)
    - 'exponential': 指数递进 (前期易，后期难)
    - 'curriculum': 课程学习 (优化版本 - 更快升级)

    参数：
    - episode: 当前训练周期
    - max_episodes: 最大训练周期
    - difficulty_strategy: 调度策略

    返回：当前难度等级 (0-4)

    优化说明：
    - 老的课程策略在Level 0浪费12500个episode，导致样本质量差
    - 新的策略在15%时已升级到Level 1，更高效利用训练时间
    """
    progress = episode / max_episodes  # 训练进度 0.0 ~ 1.0

    if difficulty_strategy == 'linear':
        # 线性调度：均匀从0→4
        difficulty = int(progress * 4.0)
    elif difficulty_strategy == 'exponential':
        # 指数调度：前期增长慢，后期加速
        difficulty = int((progress ** 1.5) * 4.0)
    elif difficulty_strategy == 'curriculum':
        # 优化的课程学习：更快地升级难度等级
        # 目标：前期学到基础动作(2-3k steps), 然后快速升级难度
        # 0-10%: Level 0 (快速探索基础)
        # 10-30%: Level 1 (适应低难度对手)
        # 30-60%: Level 2 (标准对手，稳定学习)
        # 60-85%: Level 3 (高难度对手，策略优化)
        # 85-100%: Level 4 (精英对手，最终微调)
        if progress < 0.10:
            difficulty = 0
        elif progress < 0.30:
            difficulty = 1
        elif progress < 0.60:
            difficulty = 2
        elif progress < 0.85:
            difficulty = 3
        else:
            difficulty = 4
    else:
        # 默认线性
        difficulty = int(progress * 4.0)

    return min(difficulty, 4)  # 限制最大值为4


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()

# ============================================================================
# 优化版本：v1.0 & v1.1 奖励函数重设计
# ============================================================================

# def get_reward(info, snake_index, reward, score):
#     """
#     [v1.0] 奖励函数重设计 - 分离进食奖励和游戏结局奖励

#     改进目标：
#     - 移除距离奖励，避免"靠近豆子"vs"吃到豆子"的目标冲突
#     - 清晰的即时反馈：进食 +30分
#     - 清晰的长期目标：胜利/失败 ±100分
#     - 简洁的设计，易于学习

#     奖励结构：
#     - 进食奖励: +30 (每次进食)
#     - 游戏结局: 胜+100 / 平0 / 负-100
#     - 存活奖励: 每步 +0.1
#     - 死亡惩罚: -50
#     """
#     step_reward = np.zeros(len(snake_index))

#     for i in snake_index:
#         # 1. 进食奖励 (密集信号 - 每次进食)
#         if reward[i] > 0:
#             step_reward[i] += 30  # 进食奖励，清晰的即时反馈

#         # 2. 游戏结局奖励 (稀疏信号 - 游戏结束时)
#         if score == 1:      # 游戏结束：我方胜利
#             step_reward[i] += 100
#         elif score == 2:    # 游戏结束：我方失败
#             step_reward[i] -= 100
#         # score == 0: 平局 -> 0分（中性）

#         # 3. 存活奖励与死亡惩罚
#         if reward[i] < 0:  # 如果没死亡
#             step_reward[i] += 0.1
#         else:  # 死亡惩罚
#             step_reward[i] -= 50

#     return step_reward


def get_reward(info, snake_index, reward, score):
    """
    [v1.1] 奖励幅度平衡 - 对称化胜负奖励，游戏中期奖励也平衡

    改进点：
    - 胜/负对称：±100分
    - 游戏中期：领先+30 / 平0 / 落后-30 (对称且幅度大)
    - 进食奖励保持：+30分
    - 总体设计更平衡，避免偏向某些目标

    奖励结构：
    - 进食奖励: +30 (每次进食)
    - 游戏结束: 胜+100 / 平0 / 负-100
    - 游戏进行中: 领先+30 / 平0 / 落后-30
    - 存活奖励: 每步 +0.1 (未死亡时)
    - 死亡惩罚: -50
    """
    step_reward = np.zeros(len(snake_index))

    for i in snake_index:
        # 1. 进食奖励 (密集信号)
        if reward[i] > 0:
            step_reward[i] += 30

        # 2. 游戏结局奖励 (终局稀疏信号，对称幅度)
        if score == 1:      # 游戏结束：我方胜利
            step_reward[i] += 100
        elif score == 2:    # 游戏结束：我方失败
            step_reward[i] -= 100
        # score == 0: 平局 -> 0分

        # 3. 游戏中期奖励 (进行中，对称平衡)
        elif score == 3:    # 游戏进行中：我方领先
            step_reward[i] += 30
        elif score == 4:    # 游戏进行中：我方落后
            step_reward[i] -= 30
        # 平局情况在主逻辑中处理，这里保持0分

        # 4. 存活奖励与死亡惩罚
        if reward[i] < 0:  # 没有死亡
            step_reward[i] += 0.1
        else:  # 死亡惩罚
            step_reward[i] -= 50

    return step_reward
