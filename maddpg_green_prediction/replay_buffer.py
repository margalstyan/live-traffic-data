import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, state_dims, action_dims):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.num_agents = len(state_dims)
        self.state_dims = state_dims
        self.action_dims = action_dims

        # Preallocate memory
        self.states = [np.zeros((max_size, d)) for d in state_dims]
        self.actions = [np.zeros((max_size, d)) for d in action_dims]
        self.next_states = [np.zeros((max_size, d)) for d in state_dims]
        self.rewards = [np.zeros((max_size, 1)) for _ in range(self.num_agents)]
        self.dones = [np.zeros((max_size, 1)) for _ in range(self.num_agents)]

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.states[i][self.ptr] = states[i]
            self.actions[i][self.ptr] = actions[i]
            self.rewards[i][self.ptr] = rewards[i]
            self.next_states[i][self.ptr] = next_states[i]
            self.dones[i][self.ptr] = dones[i]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)

        states = [torch.FloatTensor(s[idxs]) for s, idxs in zip(self.states, [indices] * self.num_agents)]
        actions = [torch.FloatTensor(a[idxs]) for a, idxs in zip(self.actions, [indices] * self.num_agents)]
        rewards = [torch.FloatTensor(r[idxs]) for r, idxs in zip(self.rewards, [indices] * self.num_agents)]
        next_states = [torch.FloatTensor(ns[idxs]) for ns, idxs in zip(self.next_states, [indices] * self.num_agents)]
        dones = [torch.FloatTensor(d[idxs]) for d, idxs in zip(self.dones, [indices] * self.num_agents)]

        return states, actions, rewards, next_states, dones
