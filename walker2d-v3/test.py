import os

import gymnasium as gym
from main import ComplexRewardWrapper, randomize_target
from stable_baselines3 import PPO


def test_model(model_path, num_episodes=5):
    # Ortamı oluştur
    env_id = "Humanoid-v5"
    base_env = gym.make(env_id, render_mode="human")
    target_position = randomize_target()
    env = ComplexRewardWrapper(base_env, target_position=target_position)
    
    # Modeli yükle
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if truncated:
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    # En iyi modeli test et
    MODEL_PATH = "/Users/umuttopalak/projects/artifical-intelligence/models/ppo_humanoid_trained_5000000_complex_rewards_v001"
    print(MODEL_PATH)
    test_model(MODEL_PATH)
