import tensorflow as tf
from tensorflow.keras import layers

# Input_shape: Size of image
# Action_space: Num of actions

def build_dqn_model(input_shape, action_space):
    # Building the CNN-based DQN mode for reinforcement learning

    model = tf.keras.Sequential()

    # 1st Layer:
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2))) # Reduces the spatial dimensions

    # 2nd Layer:
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2))) # Reduce 

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(action_space, activation='linear'))
    
    # Compiles the model using Mean Sqaured Error Loss for Q-Learning
    # We use Mean Squared Error because we want to minimize the difference between
    # the predicted Q-values and the tartet Q-values
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return model
