import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


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

# Select action based on epsilon-greedy policy
def select_action(state, epsilon, model):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state)
            return torch.argmax(q_values).item()

# Load the trained model
policy_net = DQN(state_dim=4, action_dim=2)  # CartPole-v1 environment için state_dim=4 ve action_dim=2
policy_net.load_state_dict(torch.load('dqn_cartpole.pth'))
policy_net.eval()  # Modeli test moduna geçiriyoruz

# Test the model
env = gym.make('CartPole-v1', render_mode ="human")

test_reward = 0
state, _ = env.reset()
state = np.array(state, dtype=np.float32)
done = False

while not done:
    action = select_action(state, 0, policy_net)  # Epsilon=0, test için
    next_state, reward, done, _, _ = env.step(action)
    next_state = np.array(next_state, dtype=np.float32)
    state = next_state
    test_reward += reward
    env.render()  # Görselleştirme

print(f"Test Reward: {test_reward}")
env.close()
