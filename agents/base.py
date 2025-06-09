import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_observation(obs):
    if isinstance(obs, dict):
        obs = obs.get("rgb", obs)
    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)
    obs = obs / 255.0
    obs = np.transpose(obs, (2, 0, 1))
    return torch.tensor(obs, dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


class BaseQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        dummy_input = torch.zeros(1, *input_shape)
        conv_out = self.conv(dummy_input)
        conv_out_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
