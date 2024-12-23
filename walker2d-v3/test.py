import os

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from stable_baselines3 import PPO


class CustomWalker2dEnv(Wrapper):
    """
    Walker2d ortamı için özel wrapper.
    Hedef hızı kontrol eder ve buna göre ödülü ayarlar.
    """
    def __init__(self, env, target_velocity=1.0):
        """
        Args:
            env: Temel ortam
            target_velocity: Hedeflenen yürüme hızı (m/s)
        """
        super().__init__(env)
        self.target_velocity = target_velocity
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Mevcut hızı al (x-ekseni boyunca)
        current_velocity = obs[8]  # Walker2d'de 8. indeks x-ekseni hızıdır
        
        # Hız farkına göre ceza uygula
        velocity_diff = abs(current_velocity - self.target_velocity)
        velocity_penalty = -0.5 * velocity_diff
        
        # Yeni ödül = orijinal ödül + hız cezası
        modified_reward = reward + velocity_penalty
        
        return obs, modified_reward, terminated, truncated, info

def test_model(model_path, num_episodes=10, target_velocity=1.0):
    """
    Eğitilmiş modeli test eder ve performansını görselleştirir.
    
    Args:
        model_path: Yüklenecek model dosyasının yolu
        num_episodes: Test edilecek episode sayısı
        target_velocity: Hedeflenen yürüme hızı (m/s)
    """
    # Ortamı oluştur
    base_env = gym.make("Walker2d-v4", render_mode="human")
    env = CustomWalker2dEnv(base_env, target_velocity=target_velocity)
    
    # Modeli yükle
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        # Episode istatistikleri
        velocities = []

        while not (done or truncated):
            # Modelden aksiyon al
            action, _ = model.predict(obs, deterministic=True)
            # Aksiyonu uygula
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            # Hızı kaydet
            velocities.append(obs[8])
        
        avg_velocity = np.mean(velocities)
        print(f"Episode {episode + 1}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Ortalama hız: {avg_velocity:.2f} m/s")
        print(f"  Hedef hız: {target_velocity} m/s")
    
    env.close()

if __name__ == "__main__":
    # Eğitilmiş modeli test et
    MODEL_PATH = "./models_subproc/denemeppo_walker2d_v4_subproc_1000000"
    
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Model bulunamadı: {MODEL_PATH}")
    else:
        print(f"Model test ediliyor: {MODEL_PATH}")
        # Hedef hızı 1.0 m/s olarak ayarla (yürüme hızı)
        test_model(MODEL_PATH, target_velocity=1.0)

