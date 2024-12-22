import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from main import ComplexRewardWrapper, randomize_target

# Sabitler
ADDITIONAL_STEPS = 1000000  # Ek eğitim adımları
MODEL_PATH = "./models/ppo_humanoid_trained_1000000_complex_rewards_v003"  # Yüklenecek model yolu
SAVE_DIR = "./models/"

def continue_training():
    # Ortamı oluştur
    env_id = "Humanoid-v5"
    base_env = gym.make(env_id)
    target_position = randomize_target()
    wrapped_env = Monitor(ComplexRewardWrapper(base_env, target_position=target_position))

    # Önceki modeli yükle
    model = PPO.load(MODEL_PATH, env=wrapped_env)
    
    # Yeni EvalCallback oluştur
    eval_callback = EvalCallback(
        wrapped_env,
        best_model_save_path='./logs/continued_training/',
        log_path='./logs/continued_training/',
        eval_freq=10000,
        deterministic=True
    )

    # Eğitime devam et
    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        callback=eval_callback,
        reset_num_timesteps=False  # Adım sayısını sıfırlama
    )

    # Yeni modeli kaydet
    continued_model_path = os.path.join(SAVE_DIR, "continued_model.zip")
    model.save(continued_model_path)
    print(f"Continued training model saved as: {continued_model_path}")

    wrapped_env.close()
    base_env.close()

if __name__ == "__main__":
    continue_training()
