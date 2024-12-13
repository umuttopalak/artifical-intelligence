import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

MAX_STEPS = 500000
TARGET_POSITION = np.array([5.0, 5.0])


def get_latest_version(directory, max_steps):
    latest_version = 0
    pattern = f"ppo_humanoid_trained_{max_steps}_complex_rewards_v"

    for filename in os.listdir(directory):
        if filename.startswith(pattern):
            try:
                version = int(filename.split('_v')[1].split('.')[0])
                latest_version = max(latest_version, version)
            except ValueError:
                print("zort")
                continue
    return latest_version


def generate_name(directory, max_steps):
    latest_version = get_latest_version(directory, max_steps)
    next_version = latest_version + 1

    name = f"ppo_humanoid_trained_{max_steps}_complex_rewards_v{next_version:03d}"
    return name


class ComplexRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_position=TARGET_POSITION, obstacles=None):
        super().__init__(env)
        self.target_position = target_position
        self.visited_positions = set()  # Keşif için kullanılan pozisyonlar
        self.obstacles = obstacles if obstacles else []  # [(x1, y1), (x2, y2), ...]

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        # Başlangıçta orijinal ödülü al
        reward = base_reward

        # 1. Hedefe Yaklaşma Ödülü
        agent_position = obs[0:2]
        distance = np.linalg.norm(agent_position - self.target_position)
        reward += (5.0 - distance)  # Mesafeye göre ödül

        # Hedefe ulaşıldığında büyük ödül
        if distance < 0.1:  # Hedefe çok yakın
            reward += 10.0

        # Hedefe yaklaşma ödülü
        prev_distance = getattr(self, 'prev_distance', distance)
        if distance < prev_distance:
            reward += 1.0  # Hedefe yaklaşıyorsa ödül
        self.prev_distance = distance

        # 2. Enerji/Eylem Büyüklüğü Cezası
        action_magnitude = np.sum(np.square(action))
        reward -= 0.03 * action_magnitude  # Eylemler büyüdükçe ceza

        # 3. Denge Ödülü
        body_tilt = np.abs(obs[2])  # Eğimi kontrol ediyoruz
        reward += max(0, 2.0 - 10 * body_tilt)

        # 4. Engel Yönetimi
        for obstacle in self.obstacles:
            obstacle_distance = np.linalg.norm(agent_position - obstacle)
            if obstacle_distance < 1.0:  # Engelle çarpışmaya çok yakın
                reward -= 10.0  # Çarpışma cezası
            elif obstacle_distance < 3.0:  # Engellere yakınlık için hafif ceza
                reward -= 1.0

        # 5. Keşif Ödülü
        current_pos = tuple(agent_position.round(1))
        if current_pos not in self.visited_positions:
            reward += 2.0  # Yeni pozisyon ödülü
            self.visited_positions.add(current_pos)

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


def main():
    env_id = "Humanoid-v4"

    obstacles = [(3.0, 3.0), (6.0, 6.0)]
    base_env = gym.make(env_id)
    wrapped_env = ComplexRewardWrapper(base_env, target_position=TARGET_POSITION, obstacles=obstacles)

    model = PPO("MlpPolicy", wrapped_env, verbose=1)
    model_name = generate_name(directory="/Users/umuttopalak/projects/artifical-intelligence/walker2d-v3/" , max_steps=MAX_STEPS)
    
    model.learn(total_timesteps=MAX_STEPS)
    wrapped_env.render()

    model.save(model_name)

    print(model_name)
    wrapped_env.close()
    base_env.close()


main()
