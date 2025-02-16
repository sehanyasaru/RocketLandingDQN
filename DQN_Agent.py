from collections import deque

import numpy as np
import torch
from numpy.random import random
from torch import optim, nn

from DQN import DQN
from Rocket import ALPHA, MEMORY_SIZE, EPSILON, ACTIONS, BATCH_SIZE, GAMMA, EPSILON_MIN, EPSILON_DECAY


class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(ACTIONS)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state_tensor)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + (GAMMA * next_q * (1 - dones))
        loss = nn.MSELoss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)