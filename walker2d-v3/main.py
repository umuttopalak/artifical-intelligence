import os

import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Sabitler
MAX_STEPS = 5000000
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
    def __init__(self, env, target_position=TARGET_POSITION):
        super().__init__(env)
        self.target_position = target_position
        self.visited_positions = set()
        self.previous_position = None
        self.steps_upright = 0
        self.steps_since_target_change = 0
        self.max_steps_before_target_change = 100
        self.target_reached_threshold = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_position = obs[0:2]
        self.steps_since_target_change = 0
        self.target_position = self._generate_new_target(obs[0:2])
        return obs, info

    def _generate_new_target(self, current_pos):
        # Mevcut konumdan minimum 5 birim uzakta yeni bir hedef oluştur
        while True:
            new_target = np.random.uniform(low=0.0, high=10.0, size=(2,))
            distance = np.linalg.norm(new_target - current_pos)
            if distance > 5.0:  # Minimum uzaklık şartı
                return new_target

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        
        reward = base_reward * 0.1
        agent_position = obs[0:2]
        
        # Hedef değiştirme kontrolü
        self.steps_since_target_change += 1
        distance_to_target = np.linalg.norm(agent_position - self.target_position)
        
        # Hedefe ulaşma kontrolü ve ödülü
        if distance_to_target < self.target_reached_threshold or self.steps_since_target_change >= self.max_steps_before_target_change:
            if distance_to_target < self.target_reached_threshold:
                reward += 5.0  # Hedefe ulaşma bonusu
            self.target_position = self._generate_new_target(agent_position)
            self.steps_since_target_change = 0
        
        # 1. Ayakta Durma ve İlerleme Ödülü
        height = obs[0]
        if height > 0.8:
            self.steps_upright += 1
            reward += 0.2
            
            # Hedefe doğru ilerleme ödülü
            if self.previous_position is not None:
                movement_vector = agent_position - self.previous_position
                direction_to_target = self.target_position - self.previous_position
                direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
                
                alignment = np.dot(movement_vector, direction_to_target)
                reward += alignment * 0.8  # Hedefe doğru hareket ödülü artırıldı
        else:
            self.steps_upright = 0
            reward -= 0.2
            
        self.previous_position = agent_position.copy()
        
        # 2. Denge Ödülü
        body_angle = obs[2]
        upright_reward = np.cos(body_angle)
        reward += upright_reward * 0.3
        
        # 3. Enerji Verimliliği
        action_penalty = np.sum(np.square(action)) * 0.0005
        reward -= action_penalty
        
        # 4. Hedefe Yaklaşma Ödülü - artırıldı
        normalized_distance = distance_to_target / np.sqrt(200)
        distance_reward = 1.0 - normalized_distance
        reward += distance_reward * 0.6  # Hedefe yaklaşma ödülü artırıldı
        
        # Erken sonlandırma
        if height < 0.3:
            done = True
            reward -= 2.0
        
        info['target_position'] = self.target_position
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

# Ana Fonksiyon
def main():
    env_id = "Humanoid-v5"

    # Dizin oluştur
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Sadece hedef pozisyonu oluştur
    target_position = randomize_target()

    # Ortamı sarmala (obstacles kaldırıldı)
    base_env = gym.make(env_id)
    wrapped_env = Monitor(ComplexRewardWrapper(base_env, target_position=target_position))

    # Hiperparametre optimizasyonu
    def optimize_agent(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.8, 0.99)

        model = PPO("MlpPolicy", wrapped_env, learning_rate=learning_rate, gamma=gamma, verbose=0)
        model.learn(total_timesteps=10000)

        eval_env = Monitor(ComplexRewardWrapper(base_env, target_position=target_position))
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
