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

train_episodes = 30000


def agent(state_shape, action_shape):
    learning_rate = 0.001

    init = tf.keras.initializers.he_uniform()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=state_shape))

    model.add(keras.layers.Conv2D(16,(2,2), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(32,(3,3), padding="same", strides=(2,2)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,(4,4), padding="same", strides=(2,2)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(128,(6,6), padding="same", strides=(3,3)))
    model.add(keras.layers.Activation("relu"))
    

    model.add(keras.layers.Flatten(name='features'))
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    print(model.summary())
    return model

def train(env, replay_memory, model, target_model, done):
    discount_factor = 0.9
    batch_size = 256
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

def heuristic(env, replay_memory, n_examples, board_shape):
    n=0
    random_threshold = 0.3
    while(n < n_examples):
        number_steps = 0
        done = False
        game_reward = 0
        observation = env.reset()[0]
        while not done:
            number_steps +=1
            random_number = np.random.rand()
            score,apple,head,tail,direction = env.get_state()
            y=head[0]
            x=head[1]
            d_y = apple[0][0] - y
            d_x = apple[0][1] - x
            if abs(d_y)>abs(d_x):#more vertical distance
                if d_y<0:
                    ddir = 0#up
                else:
                    ddir = 2#down
            else:
                if d_x<0:
                    ddir = 3#left
                else:
                    ddir = 1#right
            action = ddir-direction
            if action>1: action = 1#right
            if action<-1: action = -1#left
            n_it = 0
            while True:
                n_it += 1
                y=head[0]
                x=head[1]
                new_dir = direction + action
                if new_dir<0:
                    new_dir = 3
                elif new_dir > 3:
                    new_dir = 0

                if new_dir == 0:
                    y = y-1
                elif new_dir == 1:
                    x = x+1
                elif new_dir == 2:
                    y = y+1
                else:
                    x = x-1
                if ((y,x) in tail or x == 0 or y == 0 or x == board_shape[0]-2 or y == board_shape[0]-2) and n_it <15:
                    action = random.choices(list(actions.keys()))[0]
                else:
                    break

            # if(random_number < random_threshold):
            #     action = random.choices(list(actions.keys()))[0] 
            new_observation, reward, done, score = env.step(action) # new_observation = novo mapa
            if(reward > 0):
                replay_memory.append([observation, action, reward/number_steps, new_observation, done])
                game_reward += reward
                n+=1


def main():
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1024
    number_of_actions = 3
    env = SnakeGame(30,30,border=1)
    board_shape = (env.board.shape[0]+2*env.border,env.board.shape[1]+2*env.border,env.board.shape[2])

    model = agent(board_shape, number_of_actions)
    target_model = agent(board_shape, number_of_actions)
    target_model.set_weights(model.get_weights())
    
    replay_memory = deque(maxlen=50000)
    heuristic(env, replay_memory, 15000, board_shape)
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
                print(predicted)
                action = action[0]
            #print('chose action ' + actions[action])
            
            new_observation, reward, done, score = env.step(action) # new_observation = novo mapa
            if reward == 1:
                print("APPLE")
            replay_memory.append([observation, action, reward, new_observation, done])
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
                    print("Copying main network weights to the target network weights")
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

       
main()