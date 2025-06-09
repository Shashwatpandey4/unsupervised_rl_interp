import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from base import BaseQNetwork, ReplayBuffer, preprocess_observation, set_random_seed
from torch.optim import Adam


class DQNAgent:
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
    ):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
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

        # Main and target networks
        self.q_net = BaseQNetwork(input_shape, env.action_space.n).to(device)
        self.target_q_net = BaseQNetwork(input_shape, env.action_space.n).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
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
            self.q_net.train()
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

        # Compute current Q-values
        q_values = (
            self.q_net(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        )
        # Compute TD target using target network
        with torch.no_grad():
            next_q_values = self.target_q_net(next_obs_tensor).max(1)[0]
            target_q = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        # MSE loss
        q_loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # Periodically sync target network
        if self.steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return q_loss.item()
