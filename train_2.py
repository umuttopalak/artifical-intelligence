import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


class BalancedWalker2dEnv(Wrapper):
    """
    Walker2d ortamı için dengeli yürüme wrapper'ı.
    Tek ayak koşmayı engeller ve dengeli yürümeyi teşvik eder.
    """
    def __init__(self, env, target_velocity=3.0):
        super().__init__(env)
        self.target_velocity = target_velocity
        self.last_x_position = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x_position = obs[0]
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Mevcut hızı al
        current_velocity = obs[8]
        
        # X pozisyonu değişimini kontrol et (hareketsizlik için)
        current_x = obs[0]
        x_movement = abs(current_x - self.last_x_position)
        self.last_x_position = current_x
        
        # Hareketsizlik cezası
        stillness_penalty = -2.0 if x_movement < 0.01 else 0.0
        
        # Hız kontrolü - hedef hıza yaklaşmayı ödüllendir
        velocity_diff = abs(current_velocity - self.target_velocity)
        velocity_reward = np.exp(-1.0 * velocity_diff)  # Daha yumuşak düşüş
        
        # Minimum hız cezası
        min_velocity_penalty = -1.0 if current_velocity < 0.5 else 0.0
        
        # Ayak pozisyonları
        foot1_pos = obs[2]
        foot2_pos = obs[7]
        
        # Simetri ödülü
        foot_diff = abs(foot1_pos - foot2_pos)
        symmetry_reward = np.exp(-2.0 * foot_diff)
        
        # Yükseklik kontrolü
        height = obs[0]
        ideal_height = 1.1
        height_diff = abs(height - ideal_height)
        height_reward = np.exp(-1.0 * height_diff)
        
        # Denge kontrolü
        torso_angle = obs[1]
        balance_reward = np.exp(-1.0 * abs(torso_angle))
        
        # Enerji verimliliği
        action_penalty = -0.01 * np.sum(np.square(action))
        
        # Toplam ödül
        modified_reward = (
            1.0 * reward +           # Orijinal ödül
            0.5 * velocity_reward +  # Hız kontrolü
            0.2 * symmetry_reward +  # Simetrik yürüme
            0.2 * height_reward +    # Yükseklik kontrolü
            0.2 * balance_reward +   # Denge kontrolü
            stillness_penalty +      # Hareketsizlik cezası
            min_velocity_penalty +   # Minimum hız cezası
            action_penalty           # Minimal enerji cezası
        )
        
        return obs, modified_reward, terminated, truncated, info

# =======================
# A Y A R L A R
# =======================
ENV_ID = "Walker2d-v4"
NUM_ENVS = 8
TOTAL_STEPS = 1_000_000
LOG_DIR = "./logs_subproc_balanced/"
MODEL_DIR = "./models_subproc_balanced/"
SEED = 42

def make_env(env_id, rank, log_dir=None, seed=0):
    def _init():
        base_env = gym.make(env_id, render_mode=None)
        env = BalancedWalker2dEnv(base_env, target_velocity=3.0)
        env.reset(seed=seed + rank)
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
            env = Monitor(env, filename=monitor_path)
        
        return env
    return _init

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env_fns = [make_env(ENV_ID, rank=i, log_dir=LOG_DIR, seed=SEED) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    # PPO Model
    policy_kwargs = dict(
        net_arch=[dict(pi=[512, 512], vf=[512, 512])]  # Daha büyük ağ
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-4,        # Biraz daha yüksek öğrenme oranı
        n_steps=2048,
        batch_size=256,
        n_epochs=8,                # Daha fazla epoch
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,            # Daha yüksek clip range
        ent_coef=0.005,           # Daha az keşif
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="./tb_subproc_balanced/"
    )

    # Değerlendirme ortamı
    base_eval_env = gym.make(ENV_ID, render_mode=None)
    eval_env = BalancedWalker2dEnv(base_eval_env, target_velocity=3.0)
    eval_env = Monitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor.csv"))
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=5_000,
        n_eval_episodes=3,
        deterministic=True
    )

    # Eğitim
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # Modeli kaydet
    model_path = os.path.join(MODEL_DIR, f"balanced_walker2d_v4_{TOTAL_STEPS}")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main() 