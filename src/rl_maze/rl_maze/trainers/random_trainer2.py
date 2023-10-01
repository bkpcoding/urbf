import numpy as np
import matplotlib.pyplot as plt
import exputils as eu
from rl_maze.envs.random_sutton_maze import random_sutton_maze2
from rl_maze.envs.sutton_maze import sutton_maze
from stable_baselines3 import DQN
from rl_maze.envs.sutton_maze_env2 import Env
from rl_maze.envs.sutton_maze_env2 import Maze_XY
import torch
from gym import register
import gym
import exputils.data.logging as log
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback

def default_config():
    return eu.AttrDict(
        size = 9,
        seed = 48,
        difficulty = 1,
        timesteps = 2e5,
        net_arch = [32, 128],
        lr = 1e-4,
        gamma = 0.99,
        batch_size = 64,
        buffer_size = 100000,
        exploration_initial_eps = 1.0,
        exploration_fraction = 0.1,
        exploration_final_eps = 0.02,
        rbf_mlp = False,
        rbf_on = False,
        mrbf_on = True,
        mrbf_units = 128,
        n_neurons_per_input = 10,
        ranges = [0, 10],
        latent_dim = 32,
        sutton_maze = False,
        policy = "MLP",
    )


def run(config = None):
    wandb.init(project="rl_maze", sync_tensorboard=True)
    config = eu.combine_dicts(config, default_config())
    torch.set_num_threads(1)
    config.seed = int(config.seed*100 + config.difficulty * 100)
    # if config.sutton_maze is True then the input is matrix
    if config.sutton_maze == True:
        register(id = 'Random_Sutton-v0', entry_point = Env, max_episode_steps = 100, \
                kwargs= {'size': config.size, 'difficulty': config.difficulty, 'seed': config.seed})
        env = gym.make('Random_Sutton-v0')
        print("Printing the environment state")
        print(env.reset())
    else:
        register(id = "MazeXY-v0", entry_point = Maze_XY, max_episode_steps = 100, \
                kwargs= {'size': config.size, 'difficulty': config.difficulty, 'seed': config.seed})
        env = gym.make('MazeXY-v0')
    model = DQN('MlpPolicy', env, verbose = 1, gamma = config.gamma, learning_rate = config.lr, \
                batch_size = config.batch_size, buffer_size = config.buffer_size, \
                exploration_initial_eps = config.exploration_initial_eps, \
                exploration_fraction = config.exploration_fraction, \
                exploration_final_eps = config.exploration_final_eps, \
                config = config, learning_starts = 10000, device= 'cpu', 
                tensorboard_log = "./tensorboard/")
    print(model.policy)
    learnable_params = sum(p.numel() for p in model.policy.q_net.parameters() if p.requires_grad) 
    log.add_scalar('learnable_params', learnable_params)
    model.learn(total_timesteps = int(config.timesteps), log_interval = 1000,
                callback=WandbCallback(verbose=2))
    # evaluate the agent
    mean_reward, std_reward = evaluate_policy(model = model, env = env, n_eval_episodes = 100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    log.add_scalar('mean_reward', mean_reward)
    log.add_scalar('std_reward', std_reward)
    log.save()
    return log

if __name__ == '__main__':
    run()



