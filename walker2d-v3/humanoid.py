import os

import gym
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# # =========================
# # S A B İ T L E R
# # =========================
# MAX_STEPS = 100_0000  # Eğitim süresi
# TARGET_X = 5.0        # Hedef x konumu (tahmini)
# DT = 0.008            # Walker2d ortalama zaman adımı (her step ~0.008s)
# SAVE_DIR = "./models/"
# TENSORBOARD_DIR = "./tensorboard_logs/"

# def get_latest_version(directory, max_steps):
#     latest_version = 0
#     pattern = f"ppo_walker2d_trained_{max_steps}_complex_rewards_v"

#     if not os.path.exists(directory):
#         return latest_version

#     for filename in os.listdir(directory):
#         if filename.startswith(pattern):
#             try:
#                 version = int(filename.split('_v')[1].split('.')[0])
#                 latest_version = max(latest_version, version)
#             except ValueError:
#                 continue
#     return latest_version

# def generate_name(directory, max_steps):
#     latest_version = get_latest_version(directory, max_steps)
#     next_version = latest_version + 1
#     name = f"ppo_walker2d_trained_{max_steps}_complex_rewards_v{next_version:03d}"
#     return name

# # =========================
# #  R E W A R D   W R A P P E R
# # =========================
# class ComplexRewardWrapper(gym.Wrapper):
#     """
#     Ortam kodunu değiştirmeden, gözlem uzayındaki hızlara dayanarak
#     'tahmini x konumu' tutup reward şekillendirme yapar.
#     """
#     def __init__(self, env, target_x, dt):
#         super().__init__(env)
#         self.target_x = target_x
#         self.dt = dt
#         self.estimated_x = 0.0
#         self.previous_distance = None
#         self.last_leg = None

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)

#         # x konumunu 0 varsayıyor, reset'te sıfırlıyoruz:
#         self.estimated_x = 0.0

#         # Distance:
#         self.previous_distance = abs(self.estimated_x - self.target_x)
#         self.last_leg = None
#         return obs, info

#     def step(self, action):
#         obs, base_reward, done, truncated, info = self.env.step(action)
#         reward = base_reward

#         # ----------------------------------------------------
#         # 1) Tahmini x konumu hesaplama
#         #    Gözlemde (obs[8]) x hızına karşılık gelebilir.
#         #    Her adımda dt ile çarparak konumu arttırıyoruz.
#         # ----------------------------------------------------
#         x_vel = obs[8]  # Muhtemelen x ekseni hızı
#         self.estimated_x += x_vel * self.dt

#         distance_to_target = abs(self.estimated_x - self.target_x)

#         # Hedefe yaklaşıyorsa ödül
#         if self.previous_distance is not None:
#             if distance_to_target < self.previous_distance:
#                 reward += 1.0
#             else:
#                 reward -= 0.5

#         # ----------------------------------------------------
#         # 2) "Zıplama" cezalandırma için obs[9] (z hızı) kullanılabilir
#         # ----------------------------------------------------
#         z_vel = obs[9]  # Muhtemelen z ekseni hızı
#         if z_vel > 0.3:
#             reward -= 5.0

#         # ----------------------------------------------------
#         # 3) Örneğin bacakların foot açısı (obs[4], obs[7]) vs. 
#         #    yine obs içinden alınabilir:
#         # ----------------------------------------------------
#         foot1_angle = obs[4]
#         foot2_angle = obs[7]
#         if abs(foot1_angle) < 0.2 and abs(foot2_angle) < 0.2:
#             reward += 2.0
#         else:
#             reward -= 2.0

#         # ----------------------------------------------------
#         # 4) Adım Atma Mekaniği (aksiyon[0..2] sol, [3..5] sağ)
#         # ----------------------------------------------------
#         left_leg_torque = abs(action[0]) + abs(action[1]) + abs(action[2])
#         right_leg_torque = abs(action[3]) + abs(action[4]) + abs(action[5])

#         if self.last_leg == "right" and left_leg_torque > 1.0 and right_leg_torque < 1.0:
#             reward += 3.0
#             self.last_leg = "left"
#         elif self.last_leg == "left" and right_leg_torque > 1.0 and left_leg_torque < 1.0:
#             reward += 3.0
#             self.last_leg = "right"
#         elif self.last_leg is None:
#             # İlk adım hangi bacakla atıldı?
#             if left_leg_torque > right_leg_torque:
#                 self.last_leg = "left"
#             else:
#                 self.last_leg = "right"

#         # ----------------------------------------------------
#         # 5) Hedefe (tahmini x) ulaştıysa bölümü sonlandır
#         # ----------------------------------------------------
#         if distance_to_target < 0.5:
#             reward += 100.0
#             done = True

#         self.previous_distance = distance_to_target
#         return obs, reward, done, truncated, info

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     env_id = "Walker2d-v4"
    
#     # Eğer environment PyBullet veya custom Walker2dEnv ise,
#     # yine de gym.make(env_id) ile çağırdığınızı varsayıyoruz.
#     # Gözlemin obs[8], obs[9] vb. indexlerini print ederek doğrulayın.

#     # Klasörleri oluştur
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     os.makedirs(TENSORBOARD_DIR, exist_ok=True)

#     base_env = gym.make(env_id)
#     wrapped_env = DummyVecEnv([
#         lambda: Monitor(
#             ComplexRewardWrapper(base_env, target_x=TARGET_X, dt=DT)
#         )
#     ])

#     from stable_baselines3 import PPO

#     policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

#     model = PPO(
#         policy="MlpPolicy",
#         env=wrapped_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.98,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         clip_range_vf=0.2,
#         ent_coef=0.01,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log=TENSORBOARD_DIR,
#         device=device,
#         policy_kwargs=policy_kwargs
#     )

#     eval_callback = EvalCallback(
#         wrapped_env,
#         best_model_save_path='./logs/best_model',
#         log_path='./logs/',
#         eval_freq=5000,
#         deterministic=True,
#         n_eval_episodes=5
#     )

#     model.learn(
#         total_timesteps=MAX_STEPS,
#         progress_bar=True
#     )

#     model_name = generate_name(directory=SAVE_DIR, max_steps=MAX_STEPS)
#     model.save(os.path.join(SAVE_DIR, model_name))
#     print(f"Model saved as: {model_name}")

#     wrapped_env.close()
#     base_env.close()

# if __name__ == "__main__":
#     main()





#################################
# EĞİTİM PARAMETRELERİ
#################################
ENV_ID = "Walker2d-v4"     # Orijinal MuJoCo ortam ismi
NUM_ENVS = 4               # Kaç paralel ortam (env) oluşturmak istediğiniz
TOTAL_STEPS = 1_000_000    # Kaç adım eğitim yapılacağı
SAVE_DIR = "./models/"
LOG_DIR = "./logs/"
TENSORBOARD_DIR = "./tb/"

#################################
# ORTAMLARI OLUŞTURAN FONKSİYON
#################################
def make_env(env_id, rank, log_dir=None, seed=0):
    """
    Belirli bir ortam kopyasını oluşturup Monitor ile saran yardımcı fonksiyon.
    rank: ortamın indeks numarası
    """
    def _init():
        env = gym.make(env_id)
        if log_dir is not None:
            # Her ortama ayrı monitor dosyası
            env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
        env.seed(seed + rank)
        return env
    return _init

def main():
    # GPU mü CPU mu?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Klasörleri oluştur
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    #################################
    # 1) ÇOKLU ORTAM OLUŞTURMA
    #################################
    # SubprocVecEnv ile NUM_ENVS adet paralel ortam
    env_fns = [make_env(ENV_ID, rank=i, log_dir=LOG_DIR, seed=42) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)  # veya DummyVecEnv(env_fns) isterseniz

    #################################
    # 2) PPO MODELİNİ TANIMLA
    #################################
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
        device=device,
    )

    # Değerlendirme callback (tek bir ortamda monitor)
    # Genelde multi-environment'tan ayrı bir env ile eval yapılır,
    # ama burada basit bir şekilde aynı env'lerden birini kullanacağız.
    eval_env = gym.make(ENV_ID)  # Tek kopya
    eval_env = Monitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor.csv"))
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,     # Kaç adımda bir değerlendirme
        n_eval_episodes=5,    # Değerlendirme bölüm sayısı
        deterministic=True,
        render=False
    )

    #################################
    # 3) EĞİTİM
    #################################
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Modeli kaydet
    save_path = os.path.join(SAVE_DIR, "ppo_walker2d_v4_default")
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # Ortamları kapat
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
