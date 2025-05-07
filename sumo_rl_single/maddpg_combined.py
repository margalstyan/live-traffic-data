import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import copy
import gymnasium as gym
import traci
from sumolib import checkBinary
from torch.utils.tensorboard import SummaryWriter
import os

# === Configuration ===
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_AGENTS = 18
STATE_SIZE = 24
ACTION_SIZE = 1
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
HIDDEN_SIZE = 128
UPDATE_EVERY = 4
GLOBAL_REWARD_SCALE = 0.5
MAX_STEPS = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# === Networks ===
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE + action_size, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# === Utilities ===
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience",
            field_names=["state", "action", "reward", "next_state", "done", "global_reward"])

    def add(self, s, a, r, s2, d, gr):
        self.memory.append(self.experience(s, a, r, s2, d, gr))

    def sample(self):
        exps = random.sample(self.memory, BATCH_SIZE)
        return tuple(torch.tensor(np.vstack([getattr(e, k) for e in exps]), dtype=torch.float32).to(device)
                     for k in ["state", "action", "reward", "next_state", "done", "global_reward"])

    def __len__(self):
        return len(self.memory)


# === Agent ===
class DDPGAgent:
    def __init__(self, id, state_size, action_size):
        self.id = id
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.noise = OUNoise(action_size)
        self.memory = ReplayBuffer()
        self.t_step = 0
        self.soft_update(1.0)

        self.writer = SummaryWriter(log_dir=f"runs/agent_{self.id}")
        self.episode = 0

    def act(self, state, noise=True):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()
        if noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, s, a, r, s2, d, gr):
        self.memory.add(s, a, r, s2, d, gr)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            self.learn(*self.memory.sample())

    def learn(self, states, actions, rewards, next_states, dones, global_rewards):
        rewards += GLOBAL_REWARD_SCALE * global_rewards
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            q_targets_next = self.critic_target(next_states, target_actions)
            q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(TAU)

        # Log losses
        self.writer.add_scalar("Loss/critic", critic_loss.item(), self.episode)
        self.writer.add_scalar("Loss/actor", actor_loss.item(), self.episode)

    def soft_update(self, tau):
        for t, l in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            t.data.copy_(tau * l.data + (1 - tau) * t.data)
        for t, l in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            t.data.copy_(tau * l.data + (1 - tau) * t.data)

    def reset(self):
        self.noise.reset()

    def save_checkpoint(self, path_prefix="checkpoints"):
        os.makedirs(path_prefix, exist_ok=True)
        torch.save(self.actor_local.state_dict(), f"{path_prefix}/actor_agent_{self.id}.pth")
        torch.save(self.critic_local.state_dict(), f"{path_prefix}/critic_agent_{self.id}.pth")


# === SUMO Environment ===
class SUMOEnv:
    def __init__(self, tls_ids):
        self.tls_ids = tls_ids
        self.num_agents = len(tls_ids)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        sumo_binary = checkBinary("sumo-gui")
        traci.start([sumo_binary, "-c", "osm.sumocfg", "--no-warnings", "--start"])
        self.step_count = 0
        return np.random.randn(self.num_agents, STATE_SIZE)

    def step(self, actions):
        for i, tls in enumerate(self.tls_ids):
            phase_duration = int((actions[i][0] + 1) * 45)
            traci.trafficlight.setPhaseDuration(tls, phase_duration)

        traci.simulationStep()
        self.step_count += 1
        next_states = np.random.randn(self.num_agents, STATE_SIZE)
        local_rewards = -np.random.rand(self.num_agents)
        global_reward = -np.mean(local_rewards)
        dones = [self.step_count >= MAX_STEPS] * self.num_agents
        return next_states, local_rewards, dones, {"global_reward": global_reward}


# === Trainer ===
class MultiAgentTrainer:
    def __init__(self, tls_ids):
        self.env = SUMOEnv(tls_ids)
        self.agents = [DDPGAgent(i, STATE_SIZE, ACTION_SIZE) for i in range(NUM_AGENTS)]

    def train(self, episodes=300):
        for ep in range(episodes):
            states = self.env.reset()
            for agent in self.agents:
                agent.reset()
                agent.episode = ep

            ep_rewards = np.zeros(NUM_AGENTS)

            for t in range(MAX_STEPS):
                actions = np.array([agent.act(states[i]) for i, agent in enumerate(self.agents)])
                next_states, rewards, dones, info = self.env.step(actions)
                for i, agent in enumerate(self.agents):
                    agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i], info["global_reward"])
                    ep_rewards[i] += rewards[i]
                states = next_states
                if all(dones):
                    break

            avg_ep_reward = np.mean(ep_rewards)
            print(f"Episode {ep+1}, Avg Reward: {avg_ep_reward:.2f}")

            # Log to tensorboard
            for agent in self.agents:
                agent.writer.add_scalar("Reward/episode_avg", avg_ep_reward, ep)

            # Save checkpoints
            if (ep + 1) % 50 == 0:
                for agent in self.agents:
                    agent.save_checkpoint()


# === Main ===
if __name__ == "__main__":
    if traci.isLoaded():
        traci.close()
    sumo_binary = checkBinary("sumo-gui")
    traci.start([sumo_binary, "-c", "osm.sumocfg", "--no-warnings", "--start"])
    tls_ids = traci.trafficlight.getIDList()
    traci.close()

    print(f"Detected TLS IDs: {tls_ids}")
    trainer = MultiAgentTrainer(tls_ids)
    trainer.train()
