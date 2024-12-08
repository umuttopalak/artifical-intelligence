import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    """
    Her belirli adımda eğitilmiş modeli görsel modda test eden callback.
    """
    def __init__(self, eval_env_id, eval_freq=10000, max_steps=1000, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.eval_env_id = eval_env_id
        self.eval_freq = eval_freq
        self.max_steps = max_steps

    def _on_step(self):
        # Her eval_freq adımda bir test et
        if self.n_calls % self.eval_freq == 0:
            # Değerlendirme ortamını render modda başlat
            eval_env = gym.make(self.eval_env_id, render_mode='human')
            obs, info = eval_env.reset(seed=42)
            done = False
            step_count = 0

            while not done and step_count < self.max_steps:
                # Modelden eylemi tahmin et
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                step_count += 1
            eval_env.close()
        return True


# Eğitim ortamını oluştur (render kapalı)
env_id = "Humanoid-v4"
env = gym.make(env_id, render_mode="human")  # render_mode=None veya varsayılan

model = PPO("MlpPolicy", env, verbose=1)

# Her 50.000 adımda bir görsel test yapalım
render_callback = RenderCallback(eval_env_id=env_id, eval_freq=50000, max_steps=500)

# Modeli öğrenirken callback'i devreye sok
model.learn(total_timesteps=200000, callback=render_callback)

# Eğitim tamamlandıktan sonra model kaydedilebilir
model.save("ppo_humanoid_trained")


# import gymnasium as gym
# from stable_baselines3 import PPO

# env_id = "Humanoid-v4"
# # Dikkat: Bu yöntem çok yavaşlatır, önerilmez
# env = gym.make(env_id, render_mode='human') 

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
