import gymnasium as gym
from stable_baselines3 import PPO

# Daha önce kaydettiğiniz modeli yüklemek istediğiniz ortamı oluşturun
env_id = "Humanoid-v4"
env = gym.make(env_id)

# Kaydedilmiş modeli yükle ve env parametresi ile ortamı tekrar bağla
model = PPO.load("ppo_humanoid_trained_extended", env=env)

# Eğitimi uzatmak için ek timesteps boyunca yeniden eğit
# Örneğin 500.000 ek adım
model.learn(total_timesteps=5000000)

# İsterseniz modeli tekrar kaydedebilirsiniz
model.save("ppo_humanoid_trained_extended")
