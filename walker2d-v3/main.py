import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

# Actor ve Critic ağlarını tanımlayalım
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))  # Aksiyonları -1 ile 1 arasında sınırla

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Ortam ve parametreler
env = gym.make('Ant-v5')  # Ant-v5 ortamını yükle
state_dim = env.observation_space.shape[0]  # Durum boyutu
action_dim = env.action_space.shape[0]  # Aksiyon boyutu (sürekli aksiyonlar)

# Hiperparametreler
gamma = 0.99  # İndirim faktörü
lr_actor = 1e-4  # Actor öğrenme oranı
lr_critic = 1e-3  # Critic öğrenme oranı
tau = 0.005  # Hedef ağını güncelleme oranı
batch_size = 64

# Actor ve Critic ağlarını oluştur
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

# Hedef ağlarını oluştur
target_actor = Actor(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# Optimizer'lar
actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

# Bellek
memory = deque(maxlen=10000)
steps_done = 0

# Aksiyon seçimi (actor ağını kullanarak)
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return actor(state).squeeze(0).numpy()

# Modeli optimize etme (DDPG)
def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))
    states, actions, rewards, next_states, dones = batch

    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Critic Loss
    q_values = critic(states, actions)
    next_actions = target_actor(next_states)
    next_q_values = target_critic(next_states, next_actions)
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    critic_loss = nn.MSELoss()(q_values, expected_q_values)

    # Actor Loss
    actor_loss = -critic(states, actor(states)).mean()

    # Optimize Critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Optimize Actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft Update Target Networks
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Eğitim döngüsü
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        optimize_model()

    print(f"Episode {episode} - Total Reward: {total_reward}")

env.close()


display.stop()