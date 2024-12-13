import gymnasium as gym
from stable_baselines3 import PPO

# Kaydedilmiş modeli yükle
model = PPO.load("pp_humanoid_5M_steps")

# Ortamı başlat (render_mode='human' ile görsel çıktı alabilirsiniz)
env_id = "Humanoid-v4"
env = gym.make(env_id, render_mode='human')

# Ortamı resetle
obs, info = env.reset(seed=42)

done = False
step_count = 0
max_steps = 100000  # Kaç adım çalıştırmak istediğinizi belirleyin

while not done and step_count < max_steps:
    # Modelden eylemi tahmin et
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    step_count += 1

# Ortamı kapat
env.close()

# model = PPO.load(f"ppo_humanoid_trained_{MAX_STEPS}_for_testing")

# print("\n--- Model Dik Durma Testi Başlıyor ---\n")
# base_env = gym.make(env_id, render_mode="human")
# wrapped_env = ComplexRewardWrapper(base_env, target_position=TARGET_POSITION)

# obs, _ = wrapped_env.reset()
# for step in range(1000):  # 1000 adım boyunca test
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = wrapped_env.step(action)
#     body_tilt = np.abs(obs[2])  # Eğimi kontrol ediyoruz
#     print(f"Adım {step + 1}: Body Tilt: {body_tilt:.4f}")  # Eğim çıktısı
#     if done:
#         print("Model simülasyon sırasında devrildi.")
#         break
# print("\n--- Dik Durma Testi Tamamlandı ---\n")