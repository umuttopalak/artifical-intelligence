import os

import gymnasium as gym
import numpy as np
from main import ComplexRewardWrapper
from stable_baselines3 import PPO

TARGET_POSITION = np.array([10.0, 10.0])

env_id = "Humanoid-v5"
base_env = gym.make(env_id, render_mode="human")

# Modeli yükle
model = PPO.load('models/ppo_humanoid_trained_1000000_complex_rewards_v001')

# Ortamı sarmala
wrapped_env = ComplexRewardWrapper(
    base_env, target_position=TARGET_POSITION, obstacles=[(3.0, 3.0), (6.0, 6.0)]
)

obs, _ = wrapped_env.reset()  # İkinci dönen değeri (_info) yok say
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = wrapped_env.step(action)
    # wrapped_env.render()

    if done or truncated:
        obs, _ = wrapped_env.reset()  # İkinci dönen değeri (_info) yok say

wrapped_env.close()
base_env.close()
