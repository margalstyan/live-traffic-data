from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch
import torch.nn.functional as F

class CustomSAC(SAC):
    def train_with_buffer(self, buffer: ReplayBuffer, gradient_steps: int = 1):
        for _ in range(gradient_steps):
            # === Learning rate scheduling ===
            if callable(self.lr_schedule):
                lr = self.lr_schedule(1.0)
            else:
                lr = self.lr_schedule

            for param_group in self.policy.actor.optimizer.param_groups:
                param_group["lr"] = lr
            for param_group in self.policy.critic.optimizer.param_groups:
                param_group["lr"] = lr
            if self.ent_coef_optimizer is not None:
                for param_group in self.ent_coef_optimizer.param_groups:
                    param_group["lr"] = lr

            # === Sample a batch ===
            batch: ReplayBufferSamples = buffer.sample(self.batch_size)
            self.policy.set_training_mode(True)

            # === Compute target Q-values ===
            with torch.no_grad():
                next_actions, next_log_prob = self.policy.predict(batch.next_observations, deterministic=False)
                next_log_prob = next_log_prob.reshape(-1, 1)

                target_q1, target_q2 = self.policy.critic_target(batch.next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.ent_coef * next_log_prob
                target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * target_q

            # === Critic update ===
            current_q1, current_q2 = self.policy.critic(batch.observations, batch.actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            # === Actor and Entropy updates ===
            if self._n_updates % self.policy_delay == 0:
                actions_pi, log_prob = self.policy.predict(batch.observations, deterministic=False)
                log_prob = log_prob.reshape(-1, 1)

                q1_pi, q2_pi = self.policy.critic(batch.observations, actions_pi)
                min_q_pi = torch.min(q1_pi, q2_pi)

                actor_loss = (self.ent_coef * log_prob - min_q_pi).mean()

                self.policy.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.policy.actor.optimizer.step()

                # === Entropy coefficient update ===
                if self.ent_coef_optimizer is not None:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()
                    self.ent_coef = torch.exp(self.log_ent_coef.detach())

            self._n_updates += 1

            if self._logger:
                self._logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self._logger.record("train/critic_loss", critic_loss.item())
