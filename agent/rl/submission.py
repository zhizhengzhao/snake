import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np


HIDDEN_SIZE=256
device =  torch.device("cpu")

from typing import Union
Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': torch.nn.ReLU(),
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


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


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

            # 威胁评分
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
    state_ = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object)

    for i in agents_index:
        # [0-1] self head position
        observations[i][:2] = snakes_position[i][0][:]

        # [2-5] head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state_, width, height, head_x, head_y)
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


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='softmax'):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)
        out = self.post_dense(out)
        return out


class RLAgent(object):
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.output_activation = 'softmax'
        # 创建一个虚拟的args对象来满足Actor的要求
        class Args:
            pass
        args = Args()
        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)

    def choose_action(self, obs):
        obs = torch.Tensor([obs]).to(self.device)
        logits = self.actor(obs).cpu().detach().numpy()[0]
        return logits

    def select_action_to_env(self, obs, ctrl_index):
        logits = self.choose_action(obs)
        actions = logits2action(logits)
        action_to_env = to_joint_action(actions, ctrl_index)
        return action_to_env

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename))


def to_joint_action(action, ctrl_index):
    joint_action_ = []
    action_a = action[ctrl_index]
    each = [0] * 4
    each[action_a] = 1
    joint_action_.append(each)
    return joint_action_


def logits2action(logits):
    logits = torch.Tensor(logits).to(device)
    actions = np.array([Categorical(out).sample().item() for out in logits])
    return np.array(actions)



agent = RLAgent(30, 4, 3)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_2000.pth"
agent.load_model(actor_net)


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 30
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    o_index = obs['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    observation = get_observations(obs, indexs, obs_dim, height=board_height, width=board_width)
    actions = agent.select_action_to_env(observation, indexs.index(o_index-2))
    return actions
