import gymnasium as gym
from stable_baselines3 import PPO

# Kaydedilmiş modeli yükle
model = PPO.load("ppo_humanoid_trained_extended")

# Ortamı başlat (render_mode='human' ile görsel çıktı alabilirsiniz)
env_id = "Humanoid-v4"
env = gym.make(env_id, render_mode='human')

# Ortamı resetle
obs, info = env.reset(seed=42)

done = False
step_count = 0
max_steps = 1000  # Kaç adım çalıştırmak istediğinizi belirleyin

while not done and step_count < max_steps:
    # Modelden eylemi tahmin et
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    step_count += 1

# Ortamı kapat
env.close()
