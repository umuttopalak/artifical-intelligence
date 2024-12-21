import os
import tempfile
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

MAX_STEPS = 1000


class RenderCallback(BaseCallback):
    """
    Eğitim sırasında belirli aralıklarla modeli render eden callback.
    """

    def __init__(self, eval_env, eval_freq=10000, max_steps=MAX_STEPS, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.max_steps = max_steps

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            print(f"\nEvaluating the model at step {self.n_calls}...")
            obs, info = self.eval_env.reset(seed=42)
            done = False
            step_count = 0
            while not done and step_count < self.max_steps:
                # Modelden eylemi tahmin et
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                self.eval_env.render()
                step_count += 1
        return True
