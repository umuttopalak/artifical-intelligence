import os

import gymnasium as gym
import torch
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


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
        
        # Mevcut hızı al
        current_velocity = obs[8]
        
        # Hız farkına göre ceza uygula
        velocity_diff = abs(current_velocity - self.target_velocity)
        velocity_penalty = -1.0 * velocity_diff  # Daha sert ceza
        
        # Yeni ödül = orijinal ödül + hız cezası
        modified_reward = 2.0 * reward + velocity_penalty  # Orijinal ödülü artır
        
        return obs, modified_reward, terminated, truncated, info

# =======================
# A Y A R L A R
# =======================
ENV_ID = "Walker2d-v4"
NUM_ENVS = 8            # Paralel ortam sayısını artır
TOTAL_STEPS = 1_000_000 # 1M adım
LOG_DIR = "./logs_subproc/"
MODEL_DIR = "./models_subproc/"
SEED = 42

def make_env(env_id, rank, log_dir=None, seed=0):
    """
    Her bir ortam kopyasını oluşturan yardımcı fonksiyon.
    rank: ortamın index numarası
    """
    def _init():
        base_env = gym.make(env_id, render_mode=None)
        env = CustomWalker2dEnv(base_env, target_velocity=1.0)  # Hedef hız 1.0 m/s
        env.reset(seed=seed + rank)
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
            env = Monitor(env, filename=monitor_path)
        
        return env
    return _init

def main():
    # GPU var mı?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Klasörleri oluştur
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # =======================
    # 1) Çoklu Ortam Kurulumu
    # =======================
    env_fns = [make_env(ENV_ID, rank=i, log_dir=LOG_DIR, seed=SEED) for i in range(NUM_ENVS)]
    
    # SubprocVecEnv ile parallel ortamlarda eğitim
    vec_env = SubprocVecEnv(env_fns)

    # =======================
    # 2) PPO Model
    # =======================
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Daha küçük ağ, daha hızlı öğrenme
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,        # Daha yüksek öğrenme oranı
        n_steps=1024,              # Daha kısa rollout, daha sık güncelleme
        batch_size=128,            # Orta boy batch
        n_epochs=10,               # Daha az epoch
        gamma=0.98,                # Daha kısa vadeli ödüllere odaklan
        gae_lambda=0.9,            # Daha düşük lambda
        clip_range=0.3,            # Daha agresif güncellemeler
        ent_coef=0.01,            # Daha fazla keşif
        vf_coef=0.5,
        max_grad_norm=0.8,         # Daha yüksek gradient norm
        verbose=1,
        device=device,
        tensorboard_log="./tb_subproc/"
    )

    # =======================
    # 3) Değerlendirme Callback
    # =======================
    base_eval_env = gym.make(ENV_ID, render_mode=None)
    eval_env = CustomWalker2dEnv(base_eval_env, target_velocity=1.0)
    eval_env = Monitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor.csv"))
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=5_000,     # Daha sık değerlendirme
        n_eval_episodes=3,   # Daha az değerlendirme episode'u
        deterministic=True
    )

    # =======================
    # 4) Eğitim
    # =======================
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Modeli kaydet
    model_path = os.path.join(MODEL_DIR, f"denemeppo_walker2d_v4_subproc_{TOTAL_STEPS}")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    # Ortamları kapat
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
