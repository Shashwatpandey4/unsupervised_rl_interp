import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base import BaseQNetwork, ReplayBuffer, preprocess_observation, set_random_seed
from torch.optim import Adam


class ICMModule(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        feature_dim = 256

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically compute the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy_input)
            self.conv_out_size = conv_out.view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(self.conv_out_size, feature_dim),
            nn.ReLU(),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(), nn.Linear(256, num_actions)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def encode(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def forward(self, state, next_state, action_onehot):
        phi_state = self.encode(state)
        phi_next = self.encode(next_state)

        inverse_input = torch.cat([phi_state, phi_next], dim=1)
        pred_action = self.inverse_model(inverse_input)

        forward_input = torch.cat([phi_state, action_onehot], dim=1)
        pred_phi_next = self.forward_model(forward_input)

        return pred_action, pred_phi_next, phi_next


class ICM_Agent:
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
        beta=0.2,  # weight for forward loss and intrinsic reward scaling
    ):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        self.update_interval = update_interval
        self.target_update_freq = target_update_freq

        sample_obs = env.reset()
        if isinstance(sample_obs, dict):
            sample_obs = sample_obs["rgb"]
        if isinstance(sample_obs, torch.Tensor):
            sample_obs = sample_obs.numpy()
        if sample_obs.ndim == 4:
            sample_obs = sample_obs[0]
        input_shape = sample_obs.transpose(2, 0, 1).shape

        # Q-networks
        self.q_net = BaseQNetwork(input_shape, env.action_space.n).to(device)
        self.target_q_net = BaseQNetwork(input_shape, env.action_space.n).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # ICM module
        self.icm = ICMModule(input_shape, env.action_space.n).to(device)

        self.q_optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.icm_optimizer = Adam(self.icm.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 100000
        set_random_seed(42)

    def select_action(self, obs):
        self.steps += 1
        # Exponential epsilon decay
        eps = (
            self.eps_end
            + (self.eps_start - self.eps_end)
            * torch.exp(torch.tensor(-1.0 * self.steps / self.eps_decay)).item()
        )
        if random.random() < eps:
            return self.env.action_space.sample()
        obs_tensor = preprocess_observation(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def update(self):
        # Only update every `update_interval` steps
        if self.steps % self.update_interval != 0:
            return None

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

        # ICM predictions
        action_onehot = F.one_hot(actions_tensor, self.env.action_space.n).float()
        pred_action_logits, pred_phi_next, target_phi_next = self.icm(
            obs_tensor, next_obs_tensor, action_onehot
        )

        # ICM losses
        inverse_loss = F.cross_entropy(pred_action_logits, actions_tensor)
        forward_loss = F.mse_loss(pred_phi_next, target_phi_next.detach())
        icm_loss = inverse_loss + self.beta * forward_loss

        # Intrinsic reward
        intrinsic_reward = self.beta * (pred_phi_next - target_phi_next.detach()).pow(
            2
        ).sum(1)
        combined_rewards = rewards_tensor + intrinsic_reward

        # Q-network target
        with torch.no_grad():
            next_q_values = self.target_q_net(next_obs_tensor).max(1)[0]
            target_q = combined_rewards + self.gamma * next_q_values * (
                1 - dones_tensor
            )

        q_values = (
            self.q_net(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        )
        q_loss = F.mse_loss(q_values, target_q)

        # Gradient steps
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=0.5)
        self.q_optimizer.step()

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=0.5)
        self.icm_optimizer.step()

        # Periodically sync target Q-network
        if self.steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return q_loss.item(), icm_loss.item()
