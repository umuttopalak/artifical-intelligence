import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from train import CustomWalker2dEnv


def make_env(env_id, rank, log_dir=None, seed=0):
    def _init():
        base_env = gym.make(env_id, render_mode=None)
        env = CustomWalker2dEnv(base_env, target_velocity=1.0)
        env.reset(seed=seed + rank)
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
            env = Monitor(env, filename=monitor_path)
        
        return env
    return _init

def continue_training(model_path, additional_steps=1_000_000):
    # GPU kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Klasör yolları
    LOG_DIR = "./logs_subproc_continued/"
    MODEL_DIR = "./models_subproc_continued/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Ortam kurulumu
    env_fns = [make_env("Walker2d-v4", rank=i, log_dir=LOG_DIR, seed=42) 
               for i in range(4)]
    vec_env = SubprocVecEnv(env_fns)
    
    # Modeli yükle
    model = PPO.load(model_path, env=vec_env)
    
    # Öğrenme oranını düşür
    model.learning_rate = 1e-4
    
    # Değerlendirme ortamı
    base_eval_env = gym.make("Walker2d-v4", render_mode=None)
    eval_env = CustomWalker2dEnv(base_eval_env, target_velocity=1.0)
    eval_env = Monitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor.csv"))
    
    # Callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Eğitime devam et
    model.learn(
        total_timesteps=additional_steps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False  # Timestep sayacını sıfırlama
    )
    
    # Yeni modeli kaydet
    new_model_path = os.path.join(MODEL_DIR, 
                                 f"ppo_walker2d_continued_{additional_steps}")
    model.save(new_model_path)
    print(f"Continued model saved at: {new_model_path}")
    
    # Ortamları kapat
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    # Önceki modelin yolu
    PREVIOUS_MODEL = "./models_subproc/ppo_walker2d_v4_subproc_1000000"
    
    if not os.path.exists(PREVIOUS_MODEL + ".zip"):
        print(f"Model bulunamadı: {PREVIOUS_MODEL}")
    else:
        print(f"Eğitime devam ediliyor: {PREVIOUS_MODEL}")
        continue_training(PREVIOUS_MODEL, additional_steps=1_000_000) 