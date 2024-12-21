import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# XML oluşturma ve hedef ekleme
def create_custom_humanoid_xml(original_xml, custom_xml, target_position):
    tree = ET.parse(original_xml)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # Hedef geom ekle
    target_geom = ET.Element('geom', {
        'name': 'target',
        'type': 'sphere',
        'size': '0.2',
        'rgba': '1 0 0 1',
        'pos': f"{target_position[0]} {target_position[1]} {target_position[2]}"
    })
    worldbody.append(target_geom)

    # XML'i kaydet
    tree.write(custom_xml)
    print(f"Custom XML dosyası oluşturuldu: {custom_xml}")


# Custom ortam sınıfı
class CustomHumanoidEnv(MujocoEnv):
    def __init__(self, xml_path, target_position, render_mode=None):
        super().__init__(model_path=xml_path, frame_skip=5, render_mode=render_mode)
        self.target_position = np.array(target_position)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        # Ajanın konumu ve hedefe uzaklık
        agent_position = obs[:3]  # İlk üç eleman konum verisi olmalı
        distance_to_target = np.linalg.norm(agent_position - self.target_position)

        # Ödül hesaplama
        reward += (1.0 - distance_to_target)  # Hedefe yaklaşma ödülü
        reward -= 0.01 * np.sum(np.square(action))  # Enerji cezası

        # Denge ödülü (isteğe bağlı)
        body_tilt = np.abs(obs[2])  # Y koordinatındaki eğim
        reward += max(0, 1.0 - body_tilt)

        info['distance_to_target'] = distance_to_target
        return obs, reward, done, truncated, info


# Render callback
class RenderCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, max_steps=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.max_steps = max_steps

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            obs, info = self.eval_env.reset(seed=42)
            done = False
            step_count = 0
            while not done and step_count < self.max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                self.eval_env.render()
                step_count += 1
        return True


# Eğitim fonksiyonu
def train_custom_humanoid():
    # Sabitler
    original_xml = "humanoid.xml"
    custom_xml = "custom_humanoid.xml"
    target_position = [5.0, 5.0, 0.0]
    max_steps = 1000
    total_timesteps = 200000

    # Custom XML oluştur
    create_custom_humanoid_xml(original_xml, custom_xml, target_position)

    # Ortamları oluştur
    train_env = CustomHumanoidEnv(custom_xml, target_position=target_position, render_mode=None)
    eval_env = CustomHumanoidEnv(custom_xml, target_position=target_position, render_mode='human')

    # PPO modeli oluştur ve eğit
    model = PPO("MlpPolicy", train_env, verbose=1)
    callback = RenderCallback(eval_env=eval_env, eval_freq=50000, max_steps=max_steps)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Modeli kaydet
    model.save("ppo_custom_humanoid")
    print("Model başarıyla kaydedildi.")


if __name__ == "__main__":
    train_custom_humanoid()
