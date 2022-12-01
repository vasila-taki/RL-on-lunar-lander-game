import gym
import numpy as np
import pandas as pd
from collections import deque
import random

from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model
from tensorflow.keras.models import clone_model


import pickle
from matplotlib import pyplot as plt


class DDQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay, update_freq, batch_size):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = batch_size
        self.epsilon_min = 0.01
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        
        self.model = self.initialize_model()
        self.refresh_target_model_num = update_freq # The frequency at which the target model gets updated

    
    def initialize_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(32, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))

        # Compile the model
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        model.trainable = True

        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)

        predicted_actions = self.model.predict(state)   
        return np.argmax(predicted_actions[0])

    # save sample <s,a,r,s'> to the replay memory
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

   
    def learn_and_update_weights_by_replay(self, model_target):

        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return
        
        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 2000:
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states = self.get_attributes_from_sample(random_sample)

        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_values_state = self.model.predict_on_batch(states)
        q_values_next_state = self.model.predict_on_batch(next_states)

        # Initialize target
        target = q_values_state
        updates = np.zeros(rewards.shape)
                
        valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(self.batch_size)

        action = np.argmax(q_values_next_state, axis=1)
        q_next_state_target = self.model_target.predict_on_batch(next_states)

        updates[valid_indexes] = rewards[valid_indexes] + self.gamma * q_next_state_target[batch_indexes[valid_indexes], action[valid_indexes]]
        
        target[batch_indexes, actions] = updates
        self.model.train_on_batch(states, target)
        

    def get_attributes_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([(np.zeros(self.num_observation_space) if i[4] is None else i[3]) for i in random_sample])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return states, actions, rewards, next_states


    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample



    # default 2000 episodes
    def train(self, num_episodes=500, can_stop=True):

        # clone model
        self.model_target = clone_model(self.model)
        self.model_target.set_weights(self.model.get_weights())
        self.model_target.trainable = False
        
        for episode in range(num_episodes):
       
            state = self.env.reset()
            reward_for_episode = 0
            num_steps = 2500
            state = np.reshape(state, [1, self.num_observation_space])
            for step in range(num_steps):

                self.env.render()
                received_action = self.get_action(state)
                #print(state)
                #print("received_action:", received_action)
                next_state, reward, done, info = self.env.step(received_action)
                #print("steps reward:", reward)
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                # Store the experience in replay memory
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                # add up rewards
                reward_for_episode += reward
                state = next_state
                self.update_counter()
                self.learn_and_update_weights_by_replay(self.model_target)

                if done:
                    self.env.render()
                    print('reward for episode =', reward_for_episode)
                    break
            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each experience completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # update the target model
            if episode % self.refresh_target_model_num == 0:  
                self.model_target.set_weights(self.model.get_weights())
                print ("Target Model Refreshed")
    
            # Check for breaking condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 2000 and can_stop:
                print("model Training Complete...")
                break
            print(episode, "\t: Episode || Reward: ",reward_for_episode, "\t|| Average Reward: ",last_rewards_mean, "\t epsilon: ", self.epsilon )

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, name):
        self.model.save(name)


