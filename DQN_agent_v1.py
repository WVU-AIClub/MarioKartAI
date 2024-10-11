import random
import numpy as np
from collections import deque

import CNN_model_v1 as cnn


# Q-values are the rewards for predicting decisions/input (i.e. left, right, forward)
# Epsilon is how curious our machine is and should get less "risky" over time
# State/Next State is the frame and next frame, respectivily

class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.action_space = action_space

        # These appear to be magic numbers generated from chat-GPT
        self.memory = deque(maxlen=2000) # Replay memory to store transitions
        self.gamma = 0.95 # Discount factor for future rewards
        self.epsilon = 1.0 # Exploration rate (start fully exploring)
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay factor for epsilon
        self.batch_size = 32 # Number of experiences to sampe form memory for each training step
        self.model = cnn.build_dqn_model(input_shape, action_space)

    def remember(self, state, action, reward, next_state, done):
        # Store experiences in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Select an action: either randomly (explore) or based on Q-values (exploit).
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        q_values = self.model.predict(state) # Predicts the Q-values
        return np.argmax(q_values[0]) # Returns the action with the highest Q-value
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Use Bellman equation to update the Q-value target
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target # Update the Q-value for the action taken

            self.model.fit(state, target_f, epochs=1, verbose=0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
