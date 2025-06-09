import torch
import torch.nn as nn
import torch.nn.functional as F
from base import preprocess_observation
from torch.distributions import Categorical
from torch.optim import Adam


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class TransformerPolicy(nn.Module):
    def __init__(self, action_dim, embed_dim=128, num_layers=4, num_heads=8):
        super().__init__()
        self.encoder = CNNEncoder()
        self.flatten = nn.Flatten(2)

        self.embed_dim = embed_dim
        self.project = nn.Conv1d(64, self.embed_dim, kernel_size=1)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )

        self.fc_actor = nn.Linear(self.embed_dim, action_dim)
        self.fc_critic = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        x = self.encoder(x)  # [B, C, H, W]
        x = self.flatten(x)  # [B, C, N]
        x = self.project(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        x = self.transformer(x)
        x = x.mean(dim=1)  # aggregate over spatial tokens

        logits = self.fc_actor(x)
        value = self.fc_critic(x).squeeze(-1)
        return logits, value

    def get_attention_weights(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.project(x)
        x = x.transpose(1, 2)
        weights = []
        for layer in self.transformer.layers:
            attn = layer.self_attn
            _, attn_weights = attn(x, x, x, need_weights=True)
            weights.append(attn_weights.detach().cpu())
        return weights  # list of [B, H, N, N]

    @property
    def conv(self):
        return self.encoder.conv


class RNDModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.target = self._build_net(input_shape)
        self.predictor = self._build_net(input_shape)
        for param in self.target.parameters():
            param.requires_grad = False

    def _build_net(self, input_shape):
        cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = cnn(dummy_input)
            flat_dim = out.view(1, -1).size(1)

        self.conv = cnn

        net = nn.Sequential(
            cnn,
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        return net

    def forward(self, x):
        return self.predictor(x), self.target(x)


class TransformerRNDAgent:
    def __init__(
        self,
        env,
        device,
        gamma=0.99,
        lr=1e-4,
        eps_clip=0.2,
        update_epochs=4,
        batch_size=64,
        int_coef=2.0,
        transformer_layers=4,
        embed_dim=128,
        num_heads=8,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.int_coef = int_coef

        sample_obs = env.reset()["rgb"]
        input_shape = sample_obs[0].transpose(2, 0, 1).shape
        action_dim = env.action_space.n

        self.policy = TransformerPolicy(
            action_dim, embed_dim, transformer_layers, num_heads
        ).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        self.rnd = RNDModel(input_shape).to(device)
        self.rnd_optimizer = Adam(self.rnd.predictor.parameters(), lr=lr)

        self.memory = []

    def select_action(self, obs_batch):
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs_batch]).to(
            self.device
        )
        logits, values = self.policy(obs_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        with torch.no_grad():
            pred, target = self.rnd(obs_tensor)
            int_rewards = F.mse_loss(pred, target, reduction="none").mean(dim=1)

        return (
            actions.cpu().numpy(),
            log_probs.detach().cpu(),
            entropy.cpu(),
            values.detach().cpu(),
            int_rewards.cpu().numpy(),
        )

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        obs, actions, old_log_probs, returns, advantages, _ = zip(*self.memory)
        obs_tensor = torch.stack([preprocess_observation(o) for o in obs]).to(
            self.device
        )
        actions_tensor = torch.tensor(actions).to(self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs).to(self.device)
        returns_tensor = torch.tensor(returns).to(self.device)
        advantages_tensor = torch.tensor(advantages).to(self.device)

        rnd_loss_total = 0

        for _ in range(self.update_epochs):
            logits, values = self.policy(obs_tensor)
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

            pred, target = self.rnd(obs_tensor)
            rnd_loss = F.mse_loss(pred, target.detach())
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()

            rnd_loss_total += rnd_loss.item()

        self.memory = []
        return (
            actor_loss.item(),
            critic_loss.item(),
            rnd_loss_total / self.update_epochs,
        )
