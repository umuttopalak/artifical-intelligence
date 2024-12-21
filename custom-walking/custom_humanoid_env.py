import os
import tempfile
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from callback import RenderCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Sabitler
MAX_STEPS = 1000
TARGET_POSITION = np.array([5.0, 5.0, 0.0])  # Hedef konum (x, y, z)


class CustomHumanoidEnv(gym.Wrapper):
    def __init__(self, custom_xml_path='', target_position=TARGET_POSITION, render_mode=None):
        """
        Humanoid-v4 ortamını özelleştirerek hedef objesi ekleyen sınıf.

        :param custom_xml_path: Düzenlenmiş XML dosyasının yolu.
        :param target_position: Hedefin (x, y, z) konumu.
        :param render_mode: Render modu (örn. 'human').
        """
        super().__init__(gym.make("Humanoid-v4", xml_file=custom_xml_path, render_mode=render_mode))
        self.target_position = target_position

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        # Ödül hesaplama
        reward = base_reward

        # 1. Hedefe Yaklaşma Ödülü/Cezası
        # Humanoid gözlem yapısına göre doğru indeksleri kullanın
        # İlk üç elemanın ajan konumu olduğunu varsayıyoruz
        agent_position = obs[:3]
        distance = np.linalg.norm(agent_position - self.target_position)
        # Hedefe yaklaştıkça ödül artar, uzaklaştıkça azalır
        reward += (1.0 - distance)

        # 2. Enerji/Eylem Büyüklüğü Cezası
        action_magnitude = np.sum(np.square(action))
        reward -= 0.01 * action_magnitude  # Eylem büyüklüğüne bağlı ceza

        # 3. Denge Ödülü
        # Humanoid gözlem yapısına göre doğru indeksi kullanın
        body_tilt = np.abs(obs[2])
        reward += max(0, 1.0 - body_tilt)  # Dengeyi korudukça ödül artar

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
