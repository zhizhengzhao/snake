from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        middle_prev = [HIDDEN_SIZE, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)

        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, act_dim]
        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out = self.post_dense(out)
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE]

        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, 1]
        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch, action_batch):
        out = torch.cat((obs_batch, action_batch), dim=-1)
        out = self.prev_dense(out)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out = self.post_dense(out)
        return out


class QNetwork(nn.Module):
    """
    DQN的Q值网络

    输入: 观察 (batch_size, num_agents, obs_dim)
    输出: Q值 (batch_size, num_agents, act_dim)

    每个代理的Q值表示执行每个动作的期望回报
    """
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.args = args

        # 网络结构
        sizes_prev = [obs_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch):
        """
        Args:
            obs_batch: (batch_size, num_agents, obs_dim)

        Returns:
            q_values: (batch_size, num_agents, act_dim)
        """
        out = self.prev_dense(obs_batch)
        q_values = self.post_dense(out)
        return q_values


class LSTMNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, data, ):
        output, (_, _) = self.lstm(data)
        return output
