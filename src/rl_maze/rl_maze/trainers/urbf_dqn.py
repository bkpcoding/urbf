from glob import glob
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import numpy as np
from typing import Callable
# import video recorder from gym
from gym.wrappers.monitoring import video_recorder
from gym.envs.registration import register
import exputils as eu
import exputils.data.logging as log
from torchinfo import summary
from rl_maze.envs.maze import Env, Maze_XY, Maze_XY_Discrete, Maze_XY2, Maze_XY3, Maze_XY4, Maze_XY5, Maze_Img1, Maze_Img2, Maze_Img3
from rl_maze.envs.random_maze import Random_Maze
import random
import torch
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper, FlatObsWrapper, FullyObsWrapper, SymbolicObsWrapper
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, DummyVecEnv
import mazelab
import imageio
from matplotlib import pyplot as plt
import torch.nn as nn
import os
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
def default_config():
    return eu.AttrDict(
        seed = 42 + 3,
        gamma =  0.5,
        lr = 0.003,
        batch_size = 32,
        total_timesteps = int(500_000),
        rbf_on = False,
        rbf_mlp = False,
        n_neurons_per_input = 30,
        exp_init = 1,
        exp_fin = 0.01,
        exp_frac = 0.1,
        ranges = [0, 6],
        env = "Maze_XY4",
        deploy = True,
        mrbf_on = False,
        net_arch = [128, 64, 256],
        mrbf_units = 128,
        sutton_maze = False,
        latent_dim = 8,
        replay_buffer = False,
)
activation = {}

def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def run(config = None, **kwargs):
    #global activation
    config = eu.combine_dicts(kwargs, config, default_config())
    if config.deploy:
        torch.set_num_threads(1)
    eu.misc.seed(config.seed)
    eu.data.logging.log_to_tb = False
    sutton_dict = {"SuttonMaze": "sutton_maze", "SuttonMaze2": "sutton_maze2",
     "SuttonMaze3": "sutton_maze3", "SuttonMaze4": "sutton_maze4", "SuttonMaze5": "sutton_maze5"}
    if config.env in sutton_dict:
        config.sutton_maze = True
    if config.env == "minigrid":
        env = gym.make("MiniGrid-Empty-5x5-v0")
        env = RGBImgObsWrapper(env)
        #env = ImgObsWrapper(env)
        #env = FullyObsWrapper(env)
        #env = FlatObsWrapper(env)
        env = SymbolicObsWrapper(env)
        #env = FlatObsWrapper(env)
        env = ImgObsWrapper(env)
        # get the maximum steps of the evn
        # get the max and min of the obs space
        obs = env.reset()
        policy = "MlpPolicy"
        print("Using MiniGrid")
        #env = DummyVecEnv([lambda: env])
        #env = VecNormalize(env, norm_obs=True, norm_reward=False)
        #env = VecFrameStack(env, n_stack=4)
    elif config.env in sutton_dict:
        register(id = 'SuttonMaze-v0', entry_point = Env, max_episode_steps = 100, kwargs= {'env': sutton_dict[config.env]})
        env = gym.make('SuttonMaze-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Sutton maze")
    elif config.env == "Maze_XY":
        register(id = 'Maze_XY-v0', entry_point = Maze_XY, max_episode_steps = 100)
        env = gym.make('Maze_XY-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Maze_XY")
    elif config.env == "Maze_XY_Discrete":
        register(id = 'Maze_XY_Discrete-v0', entry_point = Maze_XY_Discrete, max_episode_steps = 10000)
        env = gym.make('Maze_XY_Discrete-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Maze_XY_Discrete")
    elif config.env == "Maze_XY2":
        register(id = 'Maze_XY2-v0', entry_point = Maze_XY2, max_episode_steps = 100)
        env = gym.make('Maze_XY2-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Maze_XY2")
    elif config.env == "RandomMaze-v0":
        register(id = 'RandomMaze-v0', entry_point = Random_Maze, max_episode_steps = 1000)
        env = gym.make('RandomMaze-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
    elif config.env == "GDY":
        import griddly
        from griddly import gd
        env = gym.make('GDY-Labyrinth-v0', level = 0, player_observer_type = gd.ObserverType.VECTOR)
        policy = 'MlpPolicy'
        print(env.observation_space)
    elif config.env == "maze2d-umaze-v1":
        import d4rl
        env = gym.make('maze2d-umaze-v1')
        policy = 'MlpPolicy'
        print(env.observation_space)
    elif config.env == "Maze_XY3":
        register(id = 'Maze_XY3-v0', entry_point = Maze_XY3, max_episode_steps = 100)
        env = gym.make('Maze_XY3-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Maze_XY3")
    elif config.env == "Maze_XY4":
        register(id = 'Maze_XY4-v0', entry_point = Maze_XY4, max_episode_steps = 100)
        env = gym.make('Maze_XY4-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Maze_XY4")
    elif config.env == "Maze_XY5":
        register(id = 'Maze_XY5-v0', entry_point = Maze_XY5, max_episode_steps = 100)
        env = gym.make('Maze_XY5-v0')
        policy = 'MlpPolicy'
        print(env.observation_space)
        print("Maze_XY5")
    elif config.env == "Maze_Img1":
        register(id = 'Maze_Img1-v0', entry_point = Maze_Img1, max_episode_steps = 100)
        env = gym.make('Maze_Img1-v0')
        # apply the openai wrapper used in nature paper 2015
        if config.rbf_on == False:
            policy = 'CnnPolicy'
        else:
            policy = "CnnRBFPolicy"
        print(env.observation_space)
        print("Maze_Img1")
    elif config.env == "Maze_Img2":
        register(id = 'Maze_Img2-v0', entry_point = Maze_Img2, max_episode_steps = 100)
        env = gym.make('Maze_Img2-v0')
        # apply the openai wrapper used in nature paper 2015
        if config.rbf_on == False:
            policy = 'CnnPolicy'
        else:
            policy = "CnnRBFPolicy"
        print(env.observation_space)
        print("Maze_Img2")
    elif config.env == "Maze_Img3":
        register(id = 'Maze_Img3-v0', entry_point = Maze_Img3, max_episode_steps = 100)
        env = gym.make('Maze_Img3-v0')
        # apply the openai wrapper used in nature paper 2015
        if config.rbf_on == False:
            policy = 'CnnPolicy'
        else:
            policy = "CnnRBFPolicy"
        print(env.observation_space)
        print("Maze_Img3")

    #if not config.deploy:
    #    log.activate_tensorboard()        
    model = DQN(policy, env, verbose = 1, learning_rate = config.lr, batch_size = config.batch_size, 
                 learning_starts=10000, buffer_size=100000, exploration_initial_eps=config.exp_init,
                  exploration_final_eps=config.exp_fin, exploration_fraction=config.exp_frac,
                 config= config, tensorboard_log=None, device= "cpu",
                )
    # log the number of parameters of the model policy
    params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    log.add_scalar("params", params)
    print("Number of parameters: ", params)

    #model = PPO(policy, env, verbose = 1, learning_rate = config.lr, batch_size = config.batch_size)

    # check if the model is saved, if so load it

#    try:
#        # print the path where load is looking for the model
#        model.set_parameters("urbf_1e6_10.zip")
#        print("Loaded model")
#    except:
#        print("No model found")
#        model.learn(total_timesteps=config.total_timesteps)
#        model.save("urbf_1e6_10")
    print(model.policy)
    # if replay buffer is used, load it
    if config.replay_buffer == True and config.env == "Maze_XY4":
        try:
            model.load_replay_buffer(f"./../../replay_buffer/Maze_XY4_True_500000.pkl")
            print("Loaded replay buffer from file")
        except:
            print("No replay buffer found")
    else:
        print("No replay buffer found")

    model.learn(total_timesteps=config.total_timesteps, log_interval=1000, eval_env = env, eval_freq = 10000)
    #save the replay buffer
    if os.path.exists("./../replay_buffer/"):
        model.save_replay_buffer(f"./../replay_buffer/{config.env}_{config.rbf_mlp}_{config.total_timesteps}_{config.net_arch}.pkl")
    else:
        os.mkdir("./../replay_buffer/")
        model.save_replay_buffer(f"./../replay_buffer/{config.env}_{config.rbf_mlp}_{config.total_timesteps}_{config.net_arch}.pkl")
     # plot the embedding after the cnn layer of the policy
    if config.env in sutton_dict and False:
        feature_extractor = nn.Sequential(model.policy.q_net.features_extractor)
        new_model = nn.Sequential(*list(model.policy.q_net.q_net[:5]))
        # use this feature extractor and new model to get the embedding for all the states
        # and plot them
        # get the states
        sutton_maze_2_state = np.array([ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1],
                                        [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
                                        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1]], dtype=np.uint8)
        embeddings = []
        sutton_maze_2_state = sutton_maze_2_state.reshape(1, 6, 14)
        for x in range(14):
            for y in range(6):
                if x >= 1 and x <= 12 and y >= 1 and y <= 5 and sutton_maze_2_state[0, y, x] == 0:
                    sutton_maze_2_state[0, y, x] = 4
                    emb = feature_extractor(torch.tensor(sutton_maze_2_state, dtype=torch.float32))
                    print(emb)
                    emb = new_model(emb)
                    embeddings.append(emb.detach().numpy())
                    sutton_maze_2_state[0, y, x] = 0
        embeddings = np.array(embeddings)
        np.set_printoptions(threshold=np.inf)
        embeddings = embeddings.reshape(38, 3)
        #hook1 = model.policy.q_net.q_net[4].register_forward_hook(getActivation('relu'))
        #out = model.policy.q_net(torch.Tensor(sutton_maze_2_state))
        # plot the embeddings
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c = "r")
        if os.path.exists("../../plots"):
            file_path = '../../plots/{}_{}_embedding.png'.format(config.env, config.seed)
        else:
            os.mkdir("../../plots")
            file_path = '../../plots/{}_{}_embedding.png'.format(config.env, config.seed)
        plt.savefig(file_path)
            
    #obs = torch.Tensor(obs)
    #cnn_out = model.policy.q_net.features_extractor.cnn(obs)
    #emb = model.policy.q_net.features_extractor.fc1(cnn_out)
    #emb = emb.detach().numpy()
    # plot the embedding
        # render the env for few steps
    if config.env == "RandomMaze-v0" or config.env == "GDY" or "Maze_XY":
        # render the environment
        video_path = f"{config.env}_video.mp4"
        vid = video_recorder.VideoRecorder(env, video_path)
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            vid.capture_frame()
            if dones:
                obs = env.reset()
    images = []
    obs = model.env.reset()
    img = model.env.render(mode = "rgb_array")
    for i in range(350):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode = "rgb_array")

    imageio.mimsave(f"{config.env}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

    
    # evaluation of the model, using the metric of success rate (1 if the agent reaches the goal, 0 otherwise)
    success = 0
    for episode in range(100):
        obs = env.reset()
        for i in range(100):
            action, _states = model.predict(obs)
            action = action.item()
            obs, rewards, dones, info = env.step(action)
            if dones and i < 99 and rewards == 0:
                success += 1
                break
    print("Success rate: ", success/100)


    if config.env == "Maze_XY" or config.env == "Maze_XY2" or config.env == "Maze_XY3" or config.env == "Maze_XY4":
        if config.env == "Maze_XY4":
            input_x = torch.tensor((0, 1, 2, 3, 4, 5, 6))
            input_y = torch.tensor((0, 1, 2, 3, 4, 5, 6))
        else:
            input_x = torch.tensor((0, 1, 2, 3, 4))
            input_y = torch.tensor((0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11))
        # input = combinations of x and y
        input = torch.stack(torch.meshgrid(input_x, input_y)).reshape(2, -1).T
        #print(input)
        # output of q_network
        output = model.q_net(input)
        #print(output)
        # plot the input and output on a heatmap
        output_1 = output[:, 0].detach().numpy()
        output_2 = output[:, 1].detach().numpy()
        output_3 = output[:, 2].detach().numpy()
        output_4 = output[:, 3].detach().numpy()
        # plot the q values of taking the above actions in each state
        fig, axs = plt.subplots(2, 2)
        #fig.colorbar(axs[0, 0].imshow(output_1.reshape(7, 7)), ax=axs[0, 0])
        #fig.colorbar(axs[0, 1].imshow(output_2.reshape(7, 7)), ax=axs[0, 1])
        #fig.colorbar(axs[1, 0].imshow(output_3.reshape(7, 7)), ax=axs[1, 0])
        #fig.colorbar(axs[1, 1].imshow(output_4.reshape(7, 7)), ax=axs[1, 1])
        # plot a single colorbar for all the subplots
        x = env.size[0]
        y = env.size[1]
        im = axs[0, 0].imshow(output_1.reshape(x, y))
        axs[0, 1].imshow(output_2.reshape(x, y))
        axs[1, 0].imshow(output_3.reshape(x, y))
        axs[1, 1].imshow(output_4.reshape(x, y))
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
        axs[0, 0].set_title('Right')
        axs[0, 1].set_title('Down')
        axs[1, 0].set_title('Left')
        axs[1, 1].set_title('Up')
        for ax in axs.flat:
            ax.set(xlabel='y', ylabel='x')
        for ax in axs.flat:
            ax.label_outer()
        #plt.show()
        # save the figure as a pdf file
        fig.savefig(f"q_values_{config.env}_{config.rbf_mlp}_{config.mrbf_on}.pdf")
        #plt.show()

    log.save()
    # play a episode and display it and record it

    #env.close()
    return log
    
