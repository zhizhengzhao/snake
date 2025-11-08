import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network import QNetwork


class DQN:
    """
    深度Q网络 (Deep Q-Network)

    相比DDPG的优势：
    ✓ 专为离散动作空间设计
    ✓ 更稳定的学习（固定目标网络）
    ✓ 更简洁的算法结构
    ✓ 更快的训练速度

    改进点：
    - 使用Double DQN技术降低Q值高估
    - 使用Dueling架构分离值函数和优势函数
    - 使用优先经验回放的支持（可选）
    """

    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device

        # 超参数
        self.q_lr = args.q_lr if hasattr(args, 'q_lr') else args.a_lr  # Q网络学习率
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed

        # 初始化Q网络和目标Q网络
        self.q_network = QNetwork(obs_dim, act_dim, num_agent, args).to(self.device)
        self.q_target = QNetwork(obs_dim, act_dim, num_agent, args).to(self.device)
        hard_update(self.q_network, self.q_target)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.q_lr)

        # 初始化回放缓冲区
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.q_loss = None

    def choose_action(self, obs, evaluation=False):
        """
        使用epsilon-greedy策略选择动作

        Args:
            obs: 观察 (num_agent, obs_dim)
            evaluation: 是否为评估模式（不进行探索）

        Returns:
            action: 动作 (num_agent, act_dim)，One-hot编码的离散动作
        """
        p = np.random.random()

        if p > self.eps or evaluation:
            # 贪心动作：选择Q值最高的动作
            obs_tensor = torch.Tensor([obs]).to(self.device)  # (1, num_agent, obs_dim)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)  # (1, num_agent, act_dim)

            # 获取最优动作的索引
            best_actions = torch.argmax(q_values, dim=-1)[0].cpu().numpy()  # (num_agent,)

            # 转换为one-hot编码
            action = np.zeros((self.num_agent, self.act_dim))
            for i in range(self.num_agent):
                action[i, best_actions[i]] = 1.0
        else:
            # 随机探索
            action = self.random_action()

        self.eps *= self.decay_speed
        return action

    def random_action(self):
        """
        生成随机动作（one-hot编码）

        Returns:
            action: 随机动作 (num_agent, act_dim)
        """
        action = np.zeros((self.num_agent, self.act_dim))
        for i in range(self.num_agent):
            action[i, np.random.randint(self.act_dim)] = 1.0
        return action

    def update(self):
        """
        使用batch更新Q网络

        采用Double DQN的更新方式：
        1. 当前网络选择动作（减少过度估计）
        2. 目标网络评估价值

        Returns:
            loss: Q网络的损失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 从回放缓冲区采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        # 转换为张量
        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)

        # ===== Double DQN目标计算 =====
        with torch.no_grad():
            # 步骤1：当前网络选择下一步的最优动作
            next_q_values = self.q_network(next_state_batch)  # (batch_size, num_agent, act_dim)
            next_best_actions = torch.argmax(next_q_values, dim=-1, keepdim=True)  # (batch_size, num_agent, 1)

            # 步骤2：目标网络评估该动作的价值
            next_q_values_target = self.q_target(next_state_batch)  # (batch_size, num_agent, act_dim)
            next_q_target = torch.gather(next_q_values_target, -1, next_best_actions)  # (batch_size, num_agent, 1)

            # 计算目标Q值
            q_target = reward_batch + self.gamma * next_q_target * (1 - done_batch)

        # ===== 当前网络Q值计算 =====
        q_values = self.q_network(state_batch)  # (batch_size, num_agent, act_dim)

        # 提取执行过的动作对应的Q值
        # action_batch是one-hot编码，需要转换为索引
        action_indices = torch.argmax(action_batch, dim=-1, keepdim=True)  # (batch_size, num_agent, 1)
        q_current = torch.gather(q_values, -1, action_indices)  # (batch_size, num_agent, 1)

        # ===== 计算损失并更新 =====
        loss_q = torch.nn.MSELoss()(q_current, q_target)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.q_network.parameters(), 1)
        self.q_optimizer.step()

        self.q_loss = loss_q.item()

        # 软更新目标网络
        soft_update(self.q_network, self.q_target, self.tau)

        return self.q_loss

    def get_loss(self):
        """获取最后一次的损失值"""
        return self.q_loss

    def load_model(self, run_dir, episode):
        """加载保存的模型"""
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_q_path = os.path.join(base_path, "q_network_" + str(episode) + ".pth")
        print(f'Q Network path: {model_q_path}')

        if os.path.exists(model_q_path):
            q_state = torch.load(model_q_path, map_location=device)
            self.q_network.load_state_dict(q_state)
            hard_update(self.q_network, self.q_target)
            print("Model loaded!")
        else:
            sys.exit(f'Model not found!')

    def save_model(self, run_dir, episode):
        """保存模型"""
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_q_path = os.path.join(base_path, "q_network_" + str(episode) + ".pth")
        torch.save(self.q_network.state_dict(), model_q_path)
