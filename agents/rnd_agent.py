import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseQNetwork, ReplayBuffer, preprocess_observation, set_random_seed
from torch.optim import Adam


class RNDModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.target = self._build_network(input_shape)
        self.predictor = self._build_network(input_shape)
        for param in self.target.parameters():
            param.requires_grad = False

    def _build_network(self, input_shape):
        conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        dummy_input = torch.zeros(1, *input_shape)
        conv_out = conv(dummy_input)
        conv_out_size = conv_out.view(1, -1).shape[1]

        return nn.Sequential(
            conv,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        target_out = self.target(x)
        pred_out = self.predictor(x)
        return pred_out, target_out


class RNDAgent:
    def __init__(
        self,
        env,
        device,
        num_envs=1,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        update_interval=4,
        target_update_freq=1000,
        int_coef=1.0,
    ):
        self.env = env
        self.device = device
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.target_update_freq = target_update_freq
        self.int_coef = int_coef

        sample_obs = env.reset()
        sample_obs = sample_obs["rgb"] if isinstance(sample_obs, dict) else sample_obs
        sample_obs = sample_obs[0] if sample_obs.ndim == 4 else sample_obs
        input_shape = sample_obs.transpose(2, 0, 1).shape

        self.q_net = BaseQNetwork(input_shape, env.action_space.n).to(device)
        self.target_q_net = BaseQNetwork(input_shape, env.action_space.n).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.rnd = RNDModel(input_shape).to(device)
        self.rnd_optimizer = Adam(self.rnd.predictor.parameters(), lr=lr)
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 100000
        set_random_seed(42)

    def compute_intrinsic_reward_batch(self, obs_batch):
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs_batch]).to(
            self.device
        )
        with torch.no_grad():
            pred = self.rnd.predictor(obs_tensor)
            target = self.rnd.target(obs_tensor)
            int_rewards = F.mse_loss(pred, target, reduction="none").mean(dim=1)
        return int_rewards.cpu().numpy()

    def select_action_batch(self, obs_batch):
        self.steps += 1
        eps = (
            self.eps_end
            + (self.eps_start - self.eps_end)
            * torch.exp(torch.tensor(-1.0 * self.steps / self.eps_decay)).item()
        )
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs_batch]).to(
            self.device
        )
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        actions = []
        for i in range(len(obs_batch)):
            if random.random() < eps:
                actions.append(self.env.action_space.sample())
            else:
                actions.append(q_values[i].argmax().item())

        if self.steps % self.update_interval == 0:
            self.update()
        if self.steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return actions

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size
        )

        obs_tensor = torch.stack([preprocess_observation(o) for o in obs]).to(
            self.device
        )
        next_obs_tensor = torch.stack([preprocess_observation(o) for o in next_obs]).to(
            self.device
        )
        actions_tensor = torch.tensor(actions).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        pred, target = self.rnd(next_obs_tensor)
        rnd_loss = F.mse_loss(pred, target.detach())
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        intrinsic_rewards = F.mse_loss(pred, target.detach(), reduction="none").mean(
            dim=1
        )
        combined_rewards = rewards_tensor + self.int_coef * intrinsic_rewards

        with torch.no_grad():
            next_q_values = self.target_q_net(next_obs_tensor).max(1)[0]
            target_q = combined_rewards + self.gamma * next_q_values * (
                1 - dones_tensor
            )

        q_values = (
            self.q_net(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        )
        q_loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        return q_loss.item(), rnd_loss.item()
