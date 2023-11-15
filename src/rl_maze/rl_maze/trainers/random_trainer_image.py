import numpy as np
import matplotlib.pyplot as plt
import exputils as eu
from rl_maze.envs.random_sutton_maze import random_sutton_maze
from stable_baselines3 import DQN
from rl_maze.envs.sutton_maze_env import Env, Maze_XY, Maze_XY_Img
import torch
from gym import register
import gym
import exputils.data.logging as log

def default_config():
    return eu.AttrDict(
        size = 8,
        seed = 0,
        difficulty = 1,
        timesteps = 2e5,
        net_arch = [],
        lr = 3e-4,
        gamma = 0.99,
        batch_size = 64,
        buffer_size = 100000,
        exploration_initial_eps = 1.0,
        exploration_fraction = 0.1,
        exploration_final_eps = 0.02,
        rbf_mlp = False,
        rbf_on = False,
        mrbf_on = True,
        mrbf_units = 64,
        n_neurons_per_input = 10,
        ranges = [0, 10],
        latent_dim = 32,
        sutton_maze = False,
        policy = "CNN"
    )

def run(config = None):
    config = eu.combine_dicts(config, default_config())
    print(config)
    torch.set_num_threads(1)
    config.seed = int(config.seed + config.difficulty * 100)
    register(id = 'Random_Sutton-v0', entry_point = Maze_XY_Img, max_episode_steps = 100, \
            kwargs= {'size': config.size, 'difficulty': config.difficulty, 'seed': config.seed})
    env = gym.make('Random_Sutton-v0')
    print(env.observation_space)
    actual_obs = env.reset()
    print("**********************")
    print(actual_obs.shape)
    print("**********************")
    if config.rbf_on == False:
        model = DQN('CnnPolicy', env, verbose = 0, gamma = config.gamma, learning_rate = config.lr, \
                    batch_size = config.batch_size, buffer_size = config.buffer_size, \
                    exploration_initial_eps = config.exploration_initial_eps, \
                    exploration_fraction = config.exploration_fraction, \
                    exploration_final_eps = config.exploration_final_eps, \
                    config = config, learning_starts = 10000, device= 'cpu')
    elif config.rbf_on == True:
        model = DQN('CnnRBFPolicy', env, verbose = 0, gamma = config.gamma, learning_rate = config.lr, \
                    batch_size = config.batch_size, buffer_size = config.buffer_size, \
                    exploration_initial_eps = config.exploration_initial_eps, \
                    exploration_fraction = config.exploration_fraction, \
                    exploration_final_eps = config.exploration_final_eps, \
                    config = config, learning_starts = 10000, device= 'cpu')
    print(model.policy)
    # count the number of parameters in the model
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    # save the number of parameters in the model in .npy 
    np.save('learnable_params.npy', total_params)
    model.learn(total_timesteps = int(config.timesteps), log_interval = 1000)
    log.save()
    return log

