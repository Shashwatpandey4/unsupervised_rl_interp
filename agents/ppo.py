import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from base import preprocess_observation, set_random_seed
from torch.distributions import Categorical
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


class PPOAgent(nn.Module):
    def __init__(
        self,
        env,
        device,
        lr=2.5e-4,
        gamma=0.99,
        eps_clip=0.2,
        update_epochs=4,
        batch_size=64,
        int_coef=0.25,
    ):
        super().__init__()
        self.env = env
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.int_coef = int_coef

        sample_obs = env.reset()
        if isinstance(sample_obs, dict):
            sample_obs = sample_obs["rgb"]
        input_shape = sample_obs[0].transpose(2, 0, 1).shape
        action_dim = env.action_space.n

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        ).to(self.device)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim + 1),
        )

        self.rnd = RNDModel(input_shape).to(self.device)
        self.rnd_optimizer = Adam(self.rnd.predictor.parameters(), lr=lr)

        self.to(self.device)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.memory = []
        set_random_seed(42)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, *shape).to(self.device)
        o = self.conv(dummy)
        return int(torch.flatten(o, 1).shape[1])

    def select_action_batch(self, obs_batch):
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs_batch]).to(
            self.device
        )
        x = self.conv(obs_tensor)
        x = torch.flatten(x, 1)
        logits_and_values = self.fc(x)
        logits = logits_and_values[:, :-1]
        values = logits_and_values[:, -1]
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.detach().cpu(), values.detach().cpu()

    def compute_intrinsic_reward(self, obs_batch):
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs_batch]).to(
            self.device
        )
        with torch.no_grad():
            pred = self.rnd.predictor(obs_tensor)
            target = self.rnd.target(obs_tensor)
            int_rewards = F.mse_loss(pred, target, reduction="none").mean(dim=1)
        return int_rewards.cpu().numpy()

    def update_rnd(self, obs_batch):
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs_batch]).to(
            self.device
        )
        pred, target = self.rnd(obs_tensor)
        rnd_loss = F.mse_loss(pred, target.detach())
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        return rnd_loss.item()

    def store(self, transition):
        self.memory.append(transition)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def update(self):
        obs, actions, old_log_probs, returns, advantages = zip(*self.memory)
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs]).to(
            self.device
        )
        actions_tensor = torch.tensor(actions).to(self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs).to(self.device)
        returns_tensor = torch.tensor(returns).to(self.device)
        advantages_tensor = torch.tensor(advantages).to(self.device)

        rnd_loss_total = 0
        for _ in range(self.update_epochs):
            x = self.conv(obs_tensor)
            x = torch.flatten(x, 1)
            logits_and_values = self.fc(x)
            logits = logits_and_values[:, :-1]
            values = logits_and_values[:, -1]
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratios * advantages_tensor
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages_tensor
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns_tensor)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            rnd_loss_total += self.update_rnd(obs)

        self.memory = []
        return (
            actor_loss.item(),
            critic_loss.item(),
            rnd_loss_total / self.update_epochs,
        )
