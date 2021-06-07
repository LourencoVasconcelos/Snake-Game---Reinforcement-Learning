# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:27:03 2021

@author: loure
"""

import tensorflow as tf
import numpy as np
from snake_game import SnakeGame
from tensorflow import keras
from collections import deque
import random

actions = {-1:'left',
            0:'same direction',
            1:'right' }

train_episodes = 300


def agent(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    print(model.summary())
    return model



def main():
    
    epsilon = 0
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1000
    number_of_actions = 3
    
    env = SnakeGame(30,30,border=1)
    board_shape = ((env.board.shape[0]+2*env.border)*(env.board.shape[1]+2*env.border)*env.board.shape[2],)

    model = agent(board_shape, number_of_actions)
    target_model = agent(board_shape, number_of_actions)
    target_model.set_weights(model.get_weights())
    
    replay_memory = deque(maxlen=50000)
    steps_to_update_target_model = 0
    
    for episodes in range(train_episodes):
         total_training_rewards = 0
         observation = env.reset()
         done = False
         while not done:
             steps_to_update_target_model += 1
             random_number = np.random.rand()
             if random_number <= epsilon:
                action = random.choices(list(actions.keys()))[0]
             else:
                #reshaped = observation.reshape([1, observation.shape[0]])
                #print(observation)
                reshaped = observation[0].reshape([1,board_shape[0]])
                predicted = model.predict(reshaped).flatten()
                action = np.argmax(predicted)-1
                
                
         new_observation, reward, done, score = env.step(action) # new_observation = novo mapa
         replay_memory.append([observation, action, reward, new_observation, done])
         
         
         if len(replay_memory) >= MIN_REPLAY_SIZE and (steps_to_update_target_model % 4 == 0 or done):
             train(env, replay_memory, model, target_model, done)
         observation = new_observation
         total_training_rewards += reward
    
    
    
    
main()