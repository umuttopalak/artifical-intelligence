import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

MAX_STEPS = 500000
TARGET_POSITION = np.array([5.0, 5.0])


class ComplexRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_position=TARGET_POSITION):
        super().__init__(env)
        self.target_position = target_position

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        reward = base_reward

        # 1. Hedefe Yaklaşma Ödülü
        agent_position = obs[0:2]
        distance = np.linalg.norm(agent_position - self.target_position)
        reward += (5.0 - distance)  # Mesafeye göre ödül

        # 2. Enerji/Eylem Büyüklüğü Cezası
        action_magnitude = np.sum(np.square(action))
        reward -= 0.03 * action_magnitude  # Eylemler büyüdükçe ceza

        # 3. Denge Ödülü
        body_tilt = np.abs(obs[2])  # Eğimi kontrol ediyoruz
        reward += max(0, 2.0 - 10 * body_tilt)
        return obs, reward, done, truncated, info


class RenderCallback(BaseCallback):
    """
    Her belirli adımda eğitilmiş modeli görsel modda test eden callback.
    """

    def __init__(self, eval_env_id, eval_freq=10000, max_steps=MAX_STEPS, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.eval_env_id = eval_env_id
        self.eval_freq = eval_freq
        self.max_steps = max_steps

    def _on_step(self):
        # Her eval_freq adımda bir test et
        if self.n_calls % self.eval_freq == 0:
            eval_env = gym.make(self.eval_env_id, render_mode='human')
            obs, info = eval_env.reset(seed=42)
            done = False
            step_count = 0

            while not done and step_count < self.max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                step_count += 1
            eval_env.close()
        return True


env_id = "Humanoid-v4"
base_env = gym.make(env_id)
wrapped_env = ComplexRewardWrapper(base_env, target_position=TARGET_POSITION)

model = PPO("MlpPolicy", wrapped_env, verbose=1)

model.learn(total_timesteps=MAX_STEPS)

model.save(f"ppo_humanoid_trained_{MAX_STEPS}_for_testing")

wrapped_env.close()
base_env.close
