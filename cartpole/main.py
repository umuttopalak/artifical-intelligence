import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 500
LEARNING_RATE = 0.001
TARGET_UPDATE = 10
NUM_EPISODES = 10000

# Gym environment
env = gym.make('CartPole-v1')

# Neural Network Model (Q-Network)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Select action based on epsilon-greedy policy
def select_action(state, epsilon, model):
    if random.random() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state)
            return torch.argmax(q_values).item()

# Initialize the Q-Networks and Optimizer
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Initialize Replay Buffer
memory = ReplayBuffer(10000)

# For tracking rewards
episode_rewards = []

# Training loop
epsilon = EPSILON_START
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    episode_reward = 0
    done = False
    
    while not done:
        action = select_action(state, epsilon, policy_net)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        
        # Store the transition in memory
        memory.push((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

        # Experience replay
        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.bool)
            
            # Compute Q targets
            with torch.no_grad():
                next_q_values = target_net(next_states)
                next_q_value = next_q_values.max(dim=1)[0]
                target_q_value = rewards + GAMMA * next_q_value * ~dones

            # Get current Q values
            q_values = policy_net(states)
            current_q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute the loss
            loss = nn.MSELoss()(current_q_value, target_q_value)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)

    episode_rewards.append(episode_reward)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 10 == 0:
        print(f"Episode {episode}/{NUM_EPISODES}, Reward: {episode_reward}")

# Modeli kaydetme
torch.save(policy_net.state_dict(), 'dqn_cartpole.pth')
print("Model başarıyla kaydedildi!")

# Plot rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward over Episodes')
plt.show()

# Test the trained agent
test_reward = 0
state, _ = env.reset()
state = np.array(state, dtype=np.float32)
done = False
while not done:
    action = select_action(state, 0, policy_net)  # Epsilon=0 for testing
    next_state, reward, done, _, _ = env.step(action)
    next_state = np.array(next_state, dtype=np.float32)
    state = next_state
    test_reward += reward
    env.render()  # Visualize the environment

print(f"Test Reward: {test_reward}")
env.close()
