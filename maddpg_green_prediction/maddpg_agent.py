import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Sigmoid()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.model(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.model(x)

class MADDPGAgent:
    def __init__(self, state_dims, action_dims, max_action=120):
        self.num_agents = len(state_dims)
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.max_action = max_action

        self.actors = [Actor(state_dims[i], action_dims[i], max_action) for i in range(self.num_agents)]
        self.critics = [Critic(sum(state_dims), sum(action_dims)) for _ in range(self.num_agents)]

        self.actor_optim = [optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        self.critic_optim = [optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]

        self.replay_buffer = ReplayBuffer(1_000_000, state_dims, action_dims)
        self.gamma = 0.95

        # TensorBoard
        self.writer = SummaryWriter(comment="-MADDPG-Traffic")
        self.training_step = 0

    def select_action(self, states):
        states = [torch.FloatTensor(s).unsqueeze(0) for s in states]
        actions = [self.actors[i](states[i]).detach().cpu().numpy().flatten() for i in range(self.num_agents)]
        return actions

    def learn(self, batch_size=128):
        if self.replay_buffer.size < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        all_states = torch.cat(states, dim=1)
        all_actions = torch.cat(actions, dim=1)
        all_next_states = torch.cat(next_states, dim=1)

        for i in range(self.num_agents):
            # Critic Update
            target_q = rewards[i] + self.gamma * self.critics[i](all_next_states, all_actions).detach() * (1 - dones[i])
            current_q = self.critics[i](all_states, all_actions)
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critic_optim[i].zero_grad()
            critic_loss.backward()
            self.critic_optim[i].step()

            # Actor Update
            current_actions = [actions[j].clone() for j in range(self.num_agents)]
            current_actions[i] = self.actors[i](states[i])
            current_actions_cat = torch.cat(current_actions, dim=1)

            actor_loss = -self.critics[i](all_states, current_actions_cat).mean()

            self.actor_optim[i].zero_grad()
            actor_loss.backward()
            self.actor_optim[i].step()

            # TensorBoard Logs
            self.writer.add_scalar(f"CriticLoss/Agent_{i}", critic_loss.item(), self.training_step)
            self.writer.add_scalar(f"ActorLoss/Agent_{i}", actor_loss.item(), self.training_step)

        self.training_step += 1
