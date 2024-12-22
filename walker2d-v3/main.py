import os
import torch

import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Sabitler
MAX_STEPS = 1000000
TARGET_POSITION = np.array([5.0, 5.0])
SAVE_DIR = "./models/"

# Yardımcı Fonksiyonlar
def get_latest_version(directory, max_steps):
    latest_version = 0
    pattern = f"ppo_humanoid_trained_{max_steps}_complex_rewards_v"

    for filename in os.listdir(directory):
        if filename.startswith(pattern):
            try:
                version = int(filename.split('_v')[1].split('.')[0])
                latest_version = max(latest_version, version)
            except ValueError:
                continue
    return latest_version

def generate_name(directory, max_steps):
    latest_version = get_latest_version(directory, max_steps)
    next_version = latest_version + 1

    name = f"ppo_humanoid_trained_{max_steps}_complex_rewards_v{next_version:03d}"
    return name

def randomize_target():
    target_position = np.random.uniform(low=0.0, high=10.0, size=(2,))
    return target_position

class ComplexRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_position):
        super().__init__(env)
        self.previous_x = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_x = obs[0]
        return obs, info

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        
        # Ortamın kendi ödül sistemini kullan ve sadece ileri hareket için ek ödül ver
        reward = base_reward
        
        # İleri hareket için bonus
        current_x = obs[0]
        forward_movement = current_x - self.previous_x
        if forward_movement > 0:
            reward += forward_movement * 0.1  # Küçük bir bonus
            
        self.previous_x = current_x
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        # Ajanın pozisyonunu çiz
        obs, _ = self.env.reset()
        agent_position = obs[0:2]
        plt.scatter(agent_position[0], agent_position[1], color="blue", label="Agent")

        # Hedef pozisyonu çiz
        plt.scatter(self.target_position[0], self.target_position[1], color="green", label="Target")

        plt.legend()
        plt.grid()
        plt.show()

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

# Ana Fonksiyon
def main():
    # GPU kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_id = "Humanoid-v5"

    # Dizin oluştur
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Ortamı oluştur
    base_env = gym.make(env_id)
    wrapped_env = Monitor(ComplexRewardWrapper(base_env, target_position=None))

    # Model parametreleri
    model = PPO(
        "MlpPolicy",
        wrapped_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,  # Keşifi açık tut
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device
    )

    # Callback
    eval_callback = EvalCallback(
        wrapped_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=5
    )

    # Eğitim
    model.learn(
        total_timesteps=1000000,
        callback=eval_callback,
        progress_bar=True
    )

    # Modeli kaydet
    model_name = generate_name(directory=SAVE_DIR, max_steps=MAX_STEPS)
    model.save(os.path.join(SAVE_DIR, model_name))
    print(f"Model saved as: {model_name}")
    
    wrapped_env.close()
    base_env.close()

if __name__ == "__main__":
    main()

