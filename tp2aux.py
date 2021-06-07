# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:51:59 2021

@author: loure
"""

import random
import tensorflow as tf
import numpy as np
from snake_game import SnakeGame
from tensorflow import keras
from collections import deque

train_episodes = 300
actions = {-1:'left',
            0:'same direction',
            1:'right' }


def agent(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.he_uniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def train(env, replay_memory, model, target_model, done):
    discount_factor = 0.618
    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        current_qs = current_qs_list[index]
        current_qs[action] = max_future_q
        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
    
def main():
    
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1000
    number_of_actions = 3
    
    env = SnakeGame(30,30,border=1)
    model = agent(env.board.shape, number_of_actions)
    
    target_model = agent(env.board.shape, number_of_actions)
    target_model.set_weights(model.get_weights())
    
    
    
    replay_memory = deque(maxlen=50000)
    steps_to_update_target_model = 0
    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            # if True:
            #     env.render()  # cagar na imagem

            
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = random.choices(list(actions.keys()))[0]
            else:
                reshaped = observation.reshape([1, observation.shape[0]]) 
                #input -> observaçao
                #board, 0, done, score
                #ver como observação chega, fazer reshape para o modelo, obter resposta 
                predicted = model.predict(reshaped).flatten()
                action = np.argmax(predicted)
                
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])
            if len(replay_memory) >= MIN_REPLAY_SIZE and (steps_to_update_target_model % 4 == 0 or done):
                train(env, replay_memory, model, target_model, done)
            observation = new_observation
            total_training_rewards += reward
            if done:
                print("Rewards: {} after n steps = {} with final reward = {}".format(total_training_rewards, episode, reward))
                total_training_rewards += 1
    
                if steps_to_update_target_model >= 100:
                    print("Copying main network weights to the target network weights")
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

main()