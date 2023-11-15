import numpy as np
import matplotlib.pyplot as plt
import exputils as eu
from rl_maze.envs.random_sutton_maze import random_sutton_maze2
from rl_maze.envs.sutton_maze import sutton_maze
from stable_baselines3 import DQN
from rl_maze.envs.sutton_maze_env import Env
from rl_maze.envs.sutton_maze_env2 import Env as Env2
import torch
from gym import register
import gym

if __name__=='__main__':
    # Eget the image from Env and show it
    for i in range(10):
        register(id = 'Random_Sutton-v0', entry_point = Env2, max_episode_steps = 100, \
                kwargs= {'size': 9, 'difficulty': 1, 'seed': (42 + i)*100 + 3 })
        try:
            env = gym.make('Random_Sutton-v0')
            env.reset()
            # save the image as a .png file
            image = env.get_image()
            print("Showing the image")
            plt.imshow(image)
            #plt.show()
            plt.savefig('second_random_sutton_maze_{}_{}.png'.format(i, 1))
        except Exception as e:
            print(e)
            print("Error in creating the environment")
            continue
        