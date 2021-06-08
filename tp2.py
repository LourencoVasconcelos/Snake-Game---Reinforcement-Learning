# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:27:03 2021

@author: loure
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.init_ops_v2 import he_uniform
from snake_game import SnakeGame
from tensorflow import keras
from collections import deque
import random
from random import choices

actions = {-1:'left',
            0:'same direction',
            1:'right' }

train_episodes = 300


def agent(state_shape, action_shape):
    learning_rate = 0.001

    init = tf.keras.initializers.he_uniform()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=state_shape))
    #Convulotional networks aqui?
    model.add(keras.layers.Flatten(name='features'))
    model.add(keras.layers.Dense(256, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    print(model.summary())
    return model

def train(env, replay_memory, model, target_model, done):
    # print(len(replay_memory))
    # print("choo choo")
    discount_factor = 0.8
    batch_size = 1024
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

def heuristic(env, replay_memory, n_examples):
    n=0
    while(n < n_examples):
        number_steps = 0
        done = False
        game_reward = 0
        observation = env.reset()[0]
        while not done:
            number_steps +=1
            random_number = np.random.rand()
            score,apple,head,tail,direction = env.get_state()
            d_row = apple[0][0] - head[0]
            d_col = apple[0][1] - head[1]
            if abs(d_row)>abs(d_col):
                if d_row<0:
                    ddir = 0
                else:
                    ddir = 2
            else:
                if d_col<0:
                    ddir = 3
                else:
                    ddir = 1
            action = ddir-direction
            if action>1: action = 1
            if action<-1: action = -1  
            new_observation, reward, done, score = env.step(action) # new_observation = novo mapa
            if(reward > 1):
                replay_memory.append([observation, action, reward, new_observation, done])
                game_reward += reward
                n+=1
                print(n)


def main():
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1024
    number_of_actions = 3
    
    env = SnakeGame(30,30,border=1, grass_growth=0, max_grass = 0.05)
    board_shape = (env.board.shape[0]+2*env.border,env.board.shape[1]+2*env.border,env.board.shape[2])

    model = agent(board_shape, number_of_actions)
    target_model = agent(board_shape, number_of_actions)
    target_model.set_weights(model.get_weights())
    
    replay_memory = deque(maxlen=50000)
    heuristic(env, replay_memory, 50000)
    steps_to_update_target_model = 0
    total_training_rewards = 0
   
    for episode in range(train_episodes):
        number_steps = 0
        observation = env.reset()[0]
        done = False
        game_reward = 0
        while not done:
            number_steps +=1
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = random.choices(list(actions.keys()))[0]
            else:
                predicted = model.predict(observation.reshape([1,*board_shape])).flatten()
                #action = np.argmax(predicted)-1
                probabilities = np.exp(predicted)/sum(np.exp(predicted))#softmax
                action = choices(list(actions.keys()), probabilities)
                #print(probabilities)
                action = action[0]
            #print('chose action ' + actions[action])
            
            new_observation, reward, done, score = env.step(action) # new_observation = novo mapa
            replay_memory.append([observation, action, reward, new_observation, done])
            print(reward)
            game_reward += reward
            #env.print_state()
            if len(replay_memory) >= MIN_REPLAY_SIZE and (steps_to_update_target_model % 4 == 0 or done):
                train(env, replay_memory, model, target_model, done)
            observation = new_observation
            total_training_rewards += reward
            if done:
                #print("Rewards: {} after n steps = {} with final reward = {}".format(total_training_rewards, episode, reward))
                print("Finished after {} steps with reward = {}".format(number_steps, game_reward))
                #total_training_rewards += 1
    
                if steps_to_update_target_model >= 100:
                    #print("Copying main network weights to the target network weights")
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

       
main()