# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:27:03 2021

@author: loure
"""
import os
import imageio

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.init_ops_v2 import he_uniform
from snake_game import SnakeGame
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import random
from random import choices

actions = {-1:'left',
            0:'same direction',
            1:'right' }

train_episodes = 3000

def plot_board(file_name,board,text=None):
    plt.figure(figsize=(10,10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45,color = 'yellow')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def agent(state_shape, action_shape):
    learning_rate = 0.001

    init = tf.keras.initializers.he_uniform()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=state_shape))

    model.add(keras.layers.Conv2D(16,(2,2), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(32,(3,3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,(4,4), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(4,4)))
    

    model.add(keras.layers.Flatten(name='features'))
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
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

def check_move(x, y, action, direction, board_shape, head, tail):
    max_x = board_shape[0]-3
    max_y = board_shape[1]-3
    new_x = x
    new_y = y
    new_direction = direction + action
    if new_direction == 4:
        new_direction = 0
    elif new_direction == -1:
        new_direction = 3
    
    if new_direction == 0:
        new_y-=1
    elif new_direction == 1:
        new_x+=1
    elif new_direction == 2:
        new_y+=1
    else:
        new_x-=1

    return (new_x >= 0 and new_x <= max_x and new_y >= 0 and new_y <= max_y and ((new_y,new_x) not in tail))
        


def heuristic(env, replay_memory, n_examples, board_shape):
    n=0
    apples = 0
    deaths = 0
    while(n < n_examples):
        number_steps = 0
        done = False
        observation = env.reset()[0]
        while not done:
            number_steps +=1
            score,apple,head,tail,direction = env.get_state()
            y,x = head
            y_apple, x_apple = apple[0]
            dist_y = y_apple-y
            dist_x = x_apple-x
            #direction (0=up, 1=right, 2=down, 3=left)
            #action -1 (turn left), 0 (continue), 1 (turn right)
            action=0
            #-y -> up
            #+y -> down
            #-x -> left
            #+x -> right
            optimal_dir = 0
            if((abs(dist_y)>abs(dist_x)) and (dist_x != 0)) or dist_y == 0:
                if(dist_x>0):
                    optimal_dir = 1
                else:
                    optimal_dir = 3
            else:
                if(dist_y>0):
                    optimal_dir = 2
                else:
                    optimal_dir = 0
            
            diff_dir = optimal_dir - direction #how many times do I turn right to get there
            if(diff_dir > 0):
                if(diff_dir == 3):
                    action = -1
                elif diff_dir==2 or diff_dir==-2:
                    possible = [-1,1]
                    action = random.choice(possible)
                else:
                    action = 1
            elif(diff_dir < 0):
                if(diff_dir == -3):
                    action = 1
                else:
                    action = -1
            moves = list(filter(lambda a: a!= action,actions.keys()))
            valid_move = check_move(x,y,action,direction,board_shape,head,tail)
            if not valid_move:
                for move in moves:
                    action = move
                    valid_move = check_move(x,y,action,direction,board_shape,head,tail)
                    if valid_move:
                        break

            new_observation, reward, done, score = env.step(action) # new_observation = novo mapa

            replay_memory.append([observation, action+1, reward, new_observation, done])
            observation = new_observation
            #plot_board("images/"+str(1000+n)+".png", observation)
            #print(n)
            n+=1


def main():
    epsilon = 0.5
    max_epsilon = 0.5
    min_epsilon = 0.01
    decay = 0.01
    MIN_REPLAY_SIZE = 1024
    number_of_actions = 3
    env = SnakeGame(30,30,border=1, max_grass = 0, food_amount = 1)
    board_shape = (env.board.shape[0]+2*env.border,env.board.shape[1]+2*env.border,env.board.shape[2])

    model = agent(board_shape, number_of_actions)
    target_model = agent(board_shape, number_of_actions)
    target_model.set_weights(model.get_weights())
    
    replay_memory = deque(maxlen=50000)
    heuristic(env, replay_memory, 50000, board_shape)

    steps_to_update_target_model = 0
    total_training_rewards = 0
    i=0
    i2=0
    for episode in range(train_episodes):
        number_steps = 0
        observation = env.reset()[0]
        done = False
        game_reward = 0
        while not done:
            # print(i)
            # if i > 5000:
            #     plot_board("images/"+str(i)+".png", observation)
            #     if i > 6000:
            #         return 0
            # i+=1
            number_steps +=1
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = random.choices(list(actions.keys()))[0]
            else:
                predicted = model.predict(observation.reshape([1,*board_shape])).flatten()
                action = np.argmax(predicted)-1
                # probabilities = np.exp(predicted)/sum(np.exp(predicted))#softmax
                # action = choices(list(actions.keys()), probabilities)
                # action = action[0]
            #print('chose action ' + actions[action])
            new_observation, reward, done, score = env.step(action) # new_observation = novo mapa
            replay_memory.append([observation, action+1, reward, new_observation, done])
            if reward >= 1 or reward == -1:
                if(reward >= 1):
                    reward =1
                game_reward += reward
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
                if(i>10000):
                    i=0
                    model.save('AP2/models/test_model_' + str(i2))
                    i2 +=1
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        print(epsilon)

def generate_gif():
    png_dir = 'images'
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('movie.gif', images)
       
main()
#generate_gif()