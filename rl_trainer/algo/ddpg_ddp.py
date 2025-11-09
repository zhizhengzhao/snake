import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update
from algo.network import Actor, Critic


class DDPG_DDP:
    """
    DDPG with Distributed Data Parallel (DDP) support for multi-GPU training.
    使用 torch.nn.parallel.DistributedDataParallel 包装 actor 和 critic 网络。

    【标准数据并行】：
    - 每张 GPU 独立采样 batch_size 的数据
    - 每张 GPU 在自己的数据上计算梯度
    - DDP 同步梯度：AllReduce + 平均
    - 有效 batch_size = batch_size × world_size
    - 学习率保持不变
    """

    def __init__(self, obs_dim, act_dim, num_agent, args, local_rank=0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.local_rank = local_rank
        self.rank = local_rank  # 当前进程的 rank
        self.device = torch.device(f"cuda:{local_rank}")

        # 获取 world_size
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # 【标准数据并行】
        # - DDP 会自动平均梯度（AllReduce + 除以 world_size）
        # - 有效 batch_size = batch_size × world_size（因为 8 卡各采样 batch_size）
        # - 学习率保持不变，DDP 会处理梯度平均
        # - 每卡采样不同数据（seed+rank），实现数据多样性
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation

        # Initialize actor network and critic network with ξ and θ
        actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)

        # Wrap models with DistributedDataParallel
        # find_unused_parameters=True 允许某些参数不被更新（如果有的话）
        self.actor = torch.nn.parallel.DistributedDataParallel(
            actor, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
        )
        self.critic = torch.nn.parallel.DistributedDataParallel(
            critic, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # Initialize target network and critic network with ξ' ← ξ and θ' ← θ
        actor_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        critic_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)

        self.actor_target = torch.nn.parallel.DistributedDataParallel(
            actor_target, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
        )
        self.critic_target = torch.nn.parallel.DistributedDataParallel(
            critic_target, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
        )

        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        # Initialize replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c_loss = None
        self.a_loss = None

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps or evaluation:
            obs = torch.Tensor([obs]).to(self.device)
            # 获取基础模块的输出
            with torch.no_grad():
                action = self.actor(obs).cpu().detach().numpy()[0]
        else:
            action = self.random_action()

        self.eps *= self.decay_speed
        return action

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    def update(self):
        """
        【标准数据并行】每卡独立采样，梯度平均

        工作原理：
        1. 每个 GPU 进程独立采样 batch_size (64) 的数据
        2. 每个进程在自己的数据上计算梯度
        3. DDP 在 backward() 时自动同步梯度：
           - AllReduce(sum)：收集所有进程的梯度
           - 除以 world_size：平均梯度
        4. 优化器更新：θ -= lr * avg_gradient

        关键特性：
        - 有效 batch_size = 64 × 8 = 512（相当于单卡 batch_size=512）
        - 数据多样性高：每卡采样不同数据（seed+rank）
        - 梯度平均确保训练稳定
        - 总计算量更大：8 卡并行计算梯度
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # 每个进程独立采样 batch_size 的数据（不共享）
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)

        # Compute target value for each agents in each transition
        with torch.no_grad():
            target_next_actions = self.actor_target(next_state_batch)
            target_next_q = self.critic_target(next_state_batch, target_next_actions)
            q_hat = reward_batch + self.gamma * target_next_q * (1 - done_batch)

        # Compute critic gradient estimation
        main_q = self.critic(state_batch, action_batch)
        loss_critic = torch.nn.MSELoss()(q_hat, main_q)

        # Update the critic networks based on Adam
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Compute actor gradient estimation
        loss_actor = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Update the actor networks based on Adam
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        # Update the target networks
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        if self.rank == 0:
            print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")

        if self.rank == 0:
            print(f'Actor path: {model_actor_path}')
            print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.device)
            critic = torch.load(model_critic_path, map_location=self.device)
            # 加载状态字典到 DDP 模型（需要使用 .module）
            self.actor.module.load_state_dict(actor)
            self.critic.module.load_state_dict(critic)
            if self.rank == 0:
                print("Model loaded!")
        else:
            if self.rank == 0:
                print(f'Model not founded at {model_actor_path} and {model_critic_path}')
            sys.exit(1)

    def save_model(self, run_dir, episode):
        # 只在 rank 0 进程保存模型，避免多个进程同时写文件
        if self.rank != 0:
            return

        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # 保存 DDP 模型的基础模块状态（.module）
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.module.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.module.state_dict(), model_critic_path)
