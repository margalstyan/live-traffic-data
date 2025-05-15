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
    def __init__(self, obs_dim, act_dim, act_limit, device="mps", writer=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        self.writer = writer
        self.train_step = 0

        # Networks
        self.actor = MLP(obs_dim, act_dim * 2).to(device)
        self.q1 = MLP(obs_dim + act_dim, 1).to(device)
        self.q2 = MLP(obs_dim + act_dim, 1).to(device)
        self.q1_target = MLP(obs_dim + act_dim, 1).to(device)
        self.q2_target = MLP(obs_dim + act_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Adaptive entropy tuning
        self.target_entropy = torch.tensor(-act_dim, dtype=torch.float32, device=device)
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True, device=device)

        # Learning rate setup
        self.lr_start = 1e-3
        self.lr_final = 1e-4
        self.lr_decay_steps = 5000  # Total steps to decay from lr_start to lr_final

        def linear_schedule(step):
            factor = max((self.lr_decay_steps - step) / self.lr_decay_steps, 0.0)
            return self.lr_final / self.lr_start + (1.0 - self.lr_final / self.lr_start) * factor

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr_start)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.lr_start)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.lr_start)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.lr_start)

        # Schedulers
        self.actor_sched = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, linear_schedule)
        self.q1_sched = torch.optim.lr_scheduler.LambdaLR(self.q1_opt, linear_schedule)
        self.q2_sched = torch.optim.lr_scheduler.LambdaLR(self.q2_opt, linear_schedule)
        self.alpha_sched = torch.optim.lr_scheduler.LambdaLR(self.alpha_opt, linear_schedule)

        # Other hyperparameters
        self.gamma = 0.99
        self.polyak = 0.995

    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        mu_logstd = self.actor(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 1)
        std = torch.exp(log_std)

        if deterministic:
            raw_action = torch.tanh(mu) * self.act_limit
        else:
            z = torch.randn_like(std)
            raw_action = torch.tanh(mu + std * z) * self.act_limit

        scaled_action = (raw_action + 1) / 2
        return scaled_action.squeeze(0)

    def update(self, replay_buffer, batch_size=256):
        batch = replay_buffer.sample_batch(batch_size)
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        act = torch.FloatTensor(batch["act"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        rew = torch.FloatTensor(batch["rew"]).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(batch["done"]).to(self.device).unsqueeze(1)

        # === Target Q computation ===
        with torch.no_grad():
            mu_logstd = self.actor(next_obs)
            mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
            log_std = torch.clamp(log_std, -5, 1)
            std = torch.exp(log_std)
            z = torch.randn_like(std)
            a_targ = torch.tanh(mu + z * std) * self.act_limit

            log_prob = -0.5 * (z**2 + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum(dim=1, keepdim=True)

            q1_t = self.q1_target(torch.cat([next_obs, a_targ], dim=-1))
            q2_t = self.q2_target(torch.cat([next_obs, a_targ], dim=-1))
            min_q_t = torch.min(q1_t, q2_t)

            alpha = self.log_alpha.exp()
            target = rew + self.gamma * (1 - done) * (min_q_t - alpha * log_prob)

        # === Critic update ===
        q1_pred = self.q1(torch.cat([obs, act], dim=-1))
        q2_pred = self.q2(torch.cat([obs, act], dim=-1))
        q1_loss = nn.MSELoss()(q1_pred, target)
        q2_loss = nn.MSELoss()(q2_pred, target)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # === Actor update ===
        mu_logstd = self.actor(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 1)
        std = torch.exp(log_std)
        z = torch.randn_like(std)
        a = torch.tanh(mu + z * std) * self.act_limit
        log_prob = -0.5 * (z**2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=1, keepdim=True)

        q1_pi = self.q1(torch.cat([obs, a], dim=-1))
        q2_pi = self.q2(torch.cat([obs, a], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_prob - min_q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # === Alpha update (entropy tuning) ===
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # === Target Q update ===
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(self.polyak).add_(p.data * (1 - self.polyak))
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(self.polyak).add_(p.data * (1 - self.polyak))

        # === TensorBoard Logging ===
        if self.writer is not None:
            with torch.no_grad():
                entropy = -log_prob.mean().item()
                mean_q1 = q1_pred.mean().item()

            self.writer.add_scalar("Loss/Q1", q1_loss.item(), self.train_step)
            self.writer.add_scalar("Loss/Q2", q2_loss.item(), self.train_step)
            self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.train_step)
            self.writer.add_scalar("Loss/Alpha", alpha_loss.item(), self.train_step)
            self.writer.add_scalar("Entropy", entropy, self.train_step)
            self.writer.add_scalar("Alpha", self.log_alpha.exp().item(), self.train_step)
            self.writer.add_scalar("LR/actor", self.actor_opt.param_groups[0]['lr'], self.train_step)
            self.writer.add_scalar("LR/q1", self.q1_opt.param_groups[0]['lr'], self.train_step)
            self.writer.add_scalar("LR/q2", self.q2_opt.param_groups[0]['lr'], self.train_step)
            self.writer.add_scalar("LR/alpha", self.alpha_opt.param_groups[0]['lr'], self.train_step)
            self.writer.add_scalar("Q1", mean_q1, self.train_step)
            self.writer.add_scalar("Q2", q2_pred.mean().item(), self.train_step)


        # === Step LR schedulers
        self.actor_sched.step()
        self.q1_sched.step()
        self.q2_sched.step()
        self.alpha_sched.step()

        self.train_step += 1


    def get_state_dicts(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
        }

    def load_state_dicts(self, state_dicts):
        self.actor.load_state_dict(state_dicts["actor"])
        self.q1.load_state_dict(state_dicts["q1"])
        self.q2.load_state_dict(state_dicts["q2"])
        self.log_alpha = torch.tensor(np.log(state_dicts["log_alpha"]), device=self.device, requires_grad=True)
