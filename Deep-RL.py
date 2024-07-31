import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

# Main training loop
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for t in range(1, 201):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

        agent.train(BATCH_SIZE)

env.close()


# actor-critic

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Hyperparameters
gamma = 0.99  # Discount factor
lr = 0.001    # Learning rate
episodes = 1000

# Create environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Instantiate networks and optimizers
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

# Training loop
for episode in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        action_probs = actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())

        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([1.0 - done])

        # Compute value of the current and next states
        value = critic(state)
        next_value = critic(next_state)

        # Compute advantage and critic loss
        advantage = reward + gamma * next_value * done - value
        critic_loss = advantage.pow(2).mean()

        # Update Critic network
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Compute actor loss
        actor_loss = -dist.log_prob(action) * advantage.detach()

        # Update Actor network
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
        total_reward += reward.item()

    print(f'Episode {episode + 1}: Total Reward = {total_reward}')

env.close()

# Deep policy gradient

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs

# Hyperparameters
gamma = 0.99  # Discount factor
lr = 0.001    # Learning rate
episodes = 1000
env_name = 'CartPole-v1'

# Create environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Instantiate the policy network and optimizer
policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)

def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Training loop
for episode in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    done = False
    rewards = []
    log_probs = []

    while not done:
        action_probs = policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state

    # Compute returns
    returns = compute_returns(rewards, gamma)
    returns = torch.FloatTensor(returns)

    # Compute policy loss
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)  # Negative sign because we want to maximize the reward

    policy_loss = torch.cat(policy_loss).sum()

    # Update the policy network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    print(f'Episode {episode + 1}: Total Reward = {total_reward}')

env.close()
