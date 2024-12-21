import os

import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
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

def randomize_target_and_obstacles():
    target_position = np.random.uniform(low=0.0, high=10.0, size=(2,))
    obstacles = [np.random.uniform(low=0.0, high=10.0, size=(2,)) for _ in range(5)]
    return target_position, obstacles

class ComplexRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_position=TARGET_POSITION, obstacles=None):
        super().__init__(env)
        self.target_position = target_position
        self.visited_positions = set()
        self.obstacles = obstacles if obstacles else []
        self.previous_position = None
        self.steps_upright = 0

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        
        # Temel ödülü koruyoruz ama ölçeklendiriyoruz
        reward = base_reward * 0.1  # Temel ödülü biraz azaltıyoruz
        
        agent_position = obs[0:2]
        
        # 1. Ayakta Durma ve İlerleme Ödülü
        height = obs[0]  # Yükseklik bilgisi
        if height > 0.8:  # Minimum yükseklik eşiği
            self.steps_upright += 1
            reward += 0.1  # Ayakta durma ödülü
            
            # İlerleme ödülü
            if self.previous_position is not None:
                forward_progress = agent_position[0] - self.previous_position[0]
                if forward_progress > 0:
                    reward += forward_progress * 0.5
        else:
            self.steps_upright = 0
            reward -= 0.1  # Düşük yükseklik cezası
            
        self.previous_position = agent_position.copy()
        
        # 2. Denge Ödülü
        body_angle = obs[2]  # Vücut açısı
        upright_reward = np.cos(body_angle)  # -1 ile 1 arası değer
        reward += upright_reward * 0.2
        
        # 3. Enerji Verimliliği
        action_penalty = np.sum(np.square(action)) * 0.001  # Eylemlerin büyüklüğü için küçük ceza
        reward -= action_penalty
        
        # 4. Hedefe Yaklaşma Ödülü (daha küçük ağırlık)
        distance_to_target = np.linalg.norm(agent_position - self.target_position)
        distance_reward = 1.0 / (1.0 + distance_to_target)  # 0 ile 1 arası normalize edilmiş
        reward += distance_reward * 0.2
        
        # 5. Engel Yönetimi (daha yumuşak cezalar)
        for obstacle in self.obstacles:
            obstacle_distance = np.linalg.norm(agent_position - obstacle)
            if obstacle_distance < 0.5:  # Çok yakın
                reward -= 0.5
            elif obstacle_distance < 1.0:  # Yakın
                reward -= 0.2
        
        # 6. Keşif Ödülü (daha küçük)
        current_pos = tuple(np.round(agent_position, decimals=1))
        if current_pos not in self.visited_positions:
            reward += 0.1
            self.visited_positions.add(current_pos)
        
        # Erken sonlandırma
        if height < 0.3:  # Çok düşük yükseklik
            done = True
            reward -= 1.0
        
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        # Render sırasında engelleri de görselleştirelim
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 10)
        plt.ylim(0, 10)

        # Ajanın pozisyonunu çiz
        obs, _ = self.env.reset()
        agent_position = obs[0:2]
        plt.scatter(agent_position[0], agent_position[1], color="blue", label="Agent")

        # Engelleri çiz
        for obstacle in self.obstacles:
            plt.scatter(obstacle[0], obstacle[1], color="red", label="Obstacle")

        # Hedef pozisyonu çiz
        plt.scatter(self.target_position[0], self.target_position[1], color="green", label="Target")

        plt.legend()
        plt.grid()
        plt.show()

# Ana Fonksiyon
def main():
    env_id = "Humanoid-v5"

    # Dizin oluştur
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Rastgele hedef ve engeller
    target_position, obstacles = randomize_target_and_obstacles()

    # Ortamı sarmala
    base_env = gym.make(env_id)
    wrapped_env = Monitor(ComplexRewardWrapper(base_env, target_position=target_position, obstacles=obstacles))

    # Hiperparametre optimizasyonu
    def optimize_agent(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.8, 0.99)

        model = PPO("MlpPolicy", wrapped_env, learning_rate=learning_rate, gamma=gamma, verbose=0)
        model.learn(total_timesteps=10000)

        eval_env = Monitor(ComplexRewardWrapper(base_env, target_position=target_position, obstacles=obstacles))
        mean_reward = []
        for _ in range(100):
            obs = eval_env.reset()
            _, reward, _, _, _ = eval_env.step(eval_env.action_space.sample())
            mean_reward.append(reward)
        return np.mean(mean_reward)

    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=20)  # Trial sayısı artırıldı

    # En iyi modelle eğitim
    model = PPO(
        "MlpPolicy",
        wrapped_env,
        **study.best_params,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    model_name = generate_name(directory=SAVE_DIR, max_steps=MAX_STEPS)

    # EvalCallback ekle
    eval_callback = EvalCallback(wrapped_env, best_model_save_path='./logs/best_model',
                                  log_path='./logs/', eval_freq=10000, deterministic=True)

    # Modeli eğit
    model.learn(total_timesteps=MAX_STEPS, callback=eval_callback)
    

    # Modeli kaydet
    model.save(os.path.join(SAVE_DIR, model_name))
    print(f"Model saved as: {model_name}")

    # Eğitim sonrası sonucu görselleştir
    wrapped_env.render()
    
    wrapped_env.close()
    base_env.close()

if __name__ == "__main__":
    main()
