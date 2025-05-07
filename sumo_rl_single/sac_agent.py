import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, device="mps"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device

        # Networks
        self.actor = MLP(obs_dim, act_dim * 2).to(device)  # mean + log_std
        self.q1 = MLP(obs_dim + act_dim, 1).to(device)
        self.q2 = MLP(obs_dim + act_dim, 1).to(device)
        self.q1_target = MLP(obs_dim + act_dim, 1).to(device)
        self.q2_target = MLP(obs_dim + act_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=3e-4)

        # Entropy
        self.alpha = 0.2
        self.gamma = 0.99
        self.polyak = 0.995

    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        mu_logstd = self.actor(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            raw_action = torch.tanh(mu) * self.act_limit
        else:
            z = torch.randn_like(std)
            raw_action = torch.tanh(mu + std * z) * self.act_limit

        scaled_action = (raw_action + 1) / 2  # Map from [-1, 1] to [0, 1]
        return scaled_action.squeeze(0)

    def update(self, replay_buffer, batch_size=256):
        batch = replay_buffer.sample_batch(batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(batch["done"]).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # Target actions
            mu_logstd = self.actor(next_obs)
            mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
            std = torch.exp(torch.clamp(log_std, -20, 2))
            z = torch.randn_like(std)
            a_targ = torch.tanh(mu + z * std) * self.act_limit
            log_prob = -0.5 * (z**2 + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum(dim=1, keepdim=True)

            q1_t = self.q1_target(torch.cat([next_obs, a_targ], dim=-1))
            q2_t = self.q2_target(torch.cat([next_obs, a_targ], dim=-1))
            min_q_t = torch.min(q1_t, q2_t)
            target = rew + self.gamma * (1 - done) * (min_q_t - self.alpha * log_prob)

        # Q1/Q2 loss
        q1_loss = nn.MSELoss()(self.q1(torch.cat([obs, act], dim=-1)), target)
        q2_loss = nn.MSELoss()(self.q2(torch.cat([obs, act], dim=-1)), target)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # Actor loss
        mu_logstd = self.actor(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        z = torch.randn_like(std)
        a = torch.tanh(mu + z * std) * self.act_limit
        log_prob = -0.5 * (z**2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=1, keepdim=True)

        q1_pi = self.q1(torch.cat([obs, a], dim=-1))
        q2_pi = self.q2(torch.cat([obs, a], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - min_q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Target update
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(self.polyak).add_(p.data * (1 - self.polyak))
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(self.polyak).add_(p.data * (1 - self.polyak))



    def get_state_dicts(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
        }

    def load_state_dicts(self, state_dicts):
        self.actor.load_state_dict(state_dicts["actor"])
        self.q1.load_state_dict(state_dicts["q1"])
        self.q2.load_state_dict(state_dicts["q2"])
