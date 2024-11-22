import gym
from gym import spaces
import numpy as np
import pyautogui
import cv2
from mss import mss
import pygetwindow as gw
import time

class MarioKartEnv(gym.Env):
    def __init__(self, window_name):
        super(MarioKartEnv, self).__init__()

        # Find and activate the game window
        self.window = gw.getWindowsWithTitle(window_name)[0]
        self.window.activate()

        self.montior = {
            "top": self.window.top,
            "left": self.window.left,
            "width": self.window.width, 
            "height": self.window.height
        }

        self.sct = mss()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.monitor['height'], self.monitor['width'], 3), dtype=np.uint8
        )

    def reset(self):
        time.sleep(1)
        return self.__get_observation()
    
    def step(self, action):

        #### NEED TO MATCH THIS TO DOPLHIN
        if action == 0:     # Accelerate
            pyautogui.keyDown('up')
        elif action == 1:   # Turn Left
            pyautogui.keyDown('left')
        elif action == 2:   # Turn Right
            pyautogui.keyDown("right")
        elif action == 3:   # Brake
            pyautogui.keyDown('down')

        time.sleep(0.1)
        pyautogui.keyUp('up')
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')
        pyautogui.keyUp('down')

        observation = self._get_observation()

        reward = self._compute_reward()

        done = self._is_done()

        return observation, reward, done, {}
    
    def _get_observation(self):
        frame = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame