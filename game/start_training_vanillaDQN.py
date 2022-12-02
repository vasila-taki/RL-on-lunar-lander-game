import os
import numpy as np
import pickle
import pandas as pd
import datetime
from matplotlib import pyplot as plt

from keras.models import load_model
# use gym reward function
from ai_mode.env import Env
#from ai_mode.simple_env import Env
from ai_mode.vanillaDQN.rl import DQN

def test_already_trained_model(trained_model):
    rewards_list = []
    num_test_episode = 100
    env = Env()
    print("Starting Testing of the trained model...")

    step_count = 2500

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode += reward
            if done:
                break
        rewards_list.append(reward_for_episode)
        print(test_episode, "\t: Episode || Reward: ", reward_for_episode)

    return rewards_list


def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
    plt.rcParams.update({'font.size': 17})
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    # plt.ylim((-4000, 4500))
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)


def plot_df2(df, chart_name, title, x_axis_label, y_axis_label):
    df['mean'] = df[df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    #plt.ylim((-4000, 4500))
    #plt.xlim((0, 100))
    plt.legend().set_visible(False)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def plot_experiments(df, chart_name, title, x_axis_label, y_axis_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1, figsize=(15, 8), title=title)
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    #plt.ylim(y_limit)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def run_experiment_for_gamma():
    print('Running Experiment for gamma...')
    env = Env()

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma_list = [0.99, 0.9, 0.8]
    training_episodes = 500
    batch_size = 64

    rewards_list_for_gammas = []
    for gamma_value in gamma_list:
        # save_dir = "hp_gamma_"+ str(gamma_value) + "_"
        model = DQN(env, lr, gamma_value, epsilon, epsilon_decay, batch_size)
        print("Training model for Gamma: {}".format(gamma_value))
        model.train(training_episodes, False)
        rewards_list_for_gammas.append(model.rewards_list)

    pickle.dump(rewards_list_for_gammas, open("rewards_list_for_gammas.p", "wb"))
    rewards_list_for_gammas = pickle.load(open("rewards_list_for_gammas.p", "rb"))

    gamma_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(gamma_list)):
        col_name = "gamma=" + str(gamma_list[i])
        gamma_rewards_pd[col_name] = rewards_list_for_gammas[i]
    plot_experiments(gamma_rewards_pd, "Figure 4: Rewards per episode for different gamma values",
                     "Figure 4: Rewards per episode for different gamma values", "Episodes", "Reward", (-4000, 4500))



def run_experiment_for_batch_size():
    print('Running Experiment for batch size...')
    env = Env()

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size_list = [32, 64, 128]
    training_episodes = 500

    rewards_list_for_batch_size = []
    for batch_size_value in batch_size_list:
        # save_dir = "hp_batch_size"+ str(batch_size_value) + "_"
        model = DQN(env, lr, gamma, epsilon, epsilon_decay, batch_size_value)
        print("Training model for batch size: {}".format(batch_size_value))
        model.train(training_episodes, False)
        rewards_list_for_batch_size.append(model.rewards_list)

    pickle.dump(rewards_list_for_batch_size, open("rewards_list_for_batch_size.p", "wb"))
    rewards_list_for_batch_size = pickle.load(open("rewards_list_for_batch_size.p", "rb"))

    batch_size_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(batch_size_list)):
        col_name = "batch_size=" + str(batch_size_list[i])
        batch_size_rewards_pd[col_name] = rewards_list_for_batch_size[i]
    plot_experiments(batch_size_rewards_pd, "Figure 4: Rewards per episode for different batch_size values",
                     "Figure 4: Rewards per episode for different batch size values", "Episodes", "Reward", (-4000, 4500))



def run_experiment_for_lr():
    print('Running Experiment for learning rate...')
    env = Env()

    # setting up params
    lr_values = [0.0001, 0.001, 0.01]
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    training_episodes = 500

    rewards_list_for_lrs = []
    for lr_value in lr_values:
        model = DQN(env, lr_value, gamma, epsilon, epsilon_decay, batch_size)
        print("Training model for LR: {}".format(lr_value))
        model.train(training_episodes, False)
        rewards_list_for_lrs.append(model.rewards_list)

    pickle.dump(rewards_list_for_lrs, open("rewards_list_for_lrs.p", "wb"))
    rewards_list_for_lrs = pickle.load(open("rewards_list_for_lrs.p", "rb"))

    lr_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes + 1)))
    for i in range(len(lr_values)):
        col_name = "lr="+ str(lr_values[i])
        lr_rewards_pd[col_name] = rewards_list_for_lrs[i]
    plot_experiments(lr_rewards_pd, "Figure 3: Rewards per episode for different learning rates", "Figure 3: Rewards per episode for different learning rates", "Episodes", "Reward", (-4000, 4500))


def run_experiment_for_ed():
    print('Running Experiment for epsilon decay...')
    env = Env() 

    # setting up params
    lr = 0.001
    epsilon = 1.0
    ed_values = [0.999, 0.995, 0.990]
    gamma = 0.99
    batch_size = 64
    training_episodes = 500

    rewards_list_for_ed = []
    for ed in ed_values:
        save_dir = "hp_ed_"+ str(ed) + "_"
        model = DQN(env, lr, gamma, epsilon, ed, batch_size)
        print("Training model for ED: {}".format(ed))
        model.train(training_episodes, False)
        rewards_list_for_ed.append(model.rewards_list)

    pickle.dump(rewards_list_for_ed, open("rewards_list_for_ed.p", "wb"))
    rewards_list_for_ed = pickle.load(open("rewards_list_for_ed.p", "rb"))

    ed_rewards_pd = pd.DataFrame(index=pd.Series(range(1, training_episodes+1)))
    for i in range(len(ed_values)):
        col_name = "epsilon_decay = "+ str(ed_values[i])
        ed_rewards_pd[col_name] = rewards_list_for_ed[i]
    plot_experiments(ed_rewards_pd, "Figure 5: Rewards per episode for different epsilon(ε) decay", "Figure 5: Rewards per episode for different epsilon(ε) decay values", "Episodes", "Reward", (-4000, 4500))


if __name__ == '__main__':
    env = Env()

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 1000
    batch_size = 128

    print('St')
    model = DQN(env, lr, gamma, epsilon, epsilon_decay, batch_size)
    
    start_time = datetime.datetime.now()
    model.train(training_episodes, True)
    print("srart of training:", start_time)
    print("end of training", datetime.datetime.now())

    # Save Everything
    save_dir = "saved_models"
    # Save trained model
    model.save(save_dir + "trained_model_simple_vanillaDQN.h5")

    # Save Rewards list
    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list_simple_vanillaDQN.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list_simple_vanillaDQN.p", "rb"))

    # plot reward in graph
    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, "Figure 1: Reward for each training episode_simple_vanillaDQN", "Reward for each training episode", "Episode","Reward")

    # Test the model
    trained_model = load_model(save_dir + "trained_model_vanillaDQN.h5")
    test_rewards = test_already_trained_model(trained_model)
    pickle.dump(test_rewards, open(save_dir + "test_rewards_vanillaDQN.p", "wb"))
    test_rewards = pickle.load(open(save_dir + "test_rewards_vanillaDQN.p", "rb"))

    plot_df2(pd.DataFrame(test_rewards), "Figure 2: Reward for each testing episode__vanillaDQN","Reward for each testing episode", "Episode", "Reward")
    print("Training and Testing Completed...!")

    # Run experiments for hyper-parameter
    #run_experiment_for_lr()
    #run_experiment_for_ed()
    #run_experiment_for_gamma()
    #run_experiment_for_batch_size()
