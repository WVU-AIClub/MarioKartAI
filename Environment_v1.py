# Here is the breakdown from GPT
# 1. State (The frame)
# 2. Action (The decisions and the execution of the action)
# 3. Rewards (Give the agent either a reward or a punishment)
# 4. Transition (?? It from one to the next)
# 5. Done (??? "The environment signals when an episode ends by returning a done flag")

import gym
import numpy as np
import cv2
import tensorflow as tf
from keras import layers

# Custom environment (agent makes the decisions based on an image)
class ImageEnv(gym.Env):
    def __init__(self):
        super(ImageEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3) # Example 3 possible actions
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.unit8)

    def reset(self):
        # Return a new image (random image or from a dataset)
        self.state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return self.state