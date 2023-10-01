import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab import BaseEnv
from mazelab import VonNeumannMotion
import gym
from gym.spaces import Box
from gym.spaces import Discrete
from rl_maze.envs.sutton_maze import sutton_maze, sutton_maze2, sutton_maze3, sutton_maze4, sutton_maze5
from gym.utils.renderer import Renderer
import cv2
from rl_maze.envs.random_sutton_maze import random_sutton_maze, check_maze, check_solvability_maze, random_sutton_maze2

class Maze(BaseMaze):
    def __init__(self, **kwargs):
        # create random maze with random_sutton_maze function and check if it is solvable
        # if not, create a new maze
        self.count = 0
        self.x = random_sutton_maze(size = kwargs['size'], difficulty = kwargs['difficulty'], seed = kwargs['seed'])
        while not check_solvability_maze(self.x):
            self.count = self.count + 1
            self.x = random_sutton_maze(size = kwargs['size'], difficulty = kwargs['difficulty'], seed = kwargs['seed'] + self.count)
            if self.count > 100:
                raise Exception('No solvable maze found')
        super().__init__(**kwargs)


    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        cliff = Object('cliff', 2, color.cliff, False, np.stack(np.where(self.x == 2), axis=1))
        goal = Object('goal', 3, color.goal, False, [])
        agent = Object('agent', 4, color.agent, False, [])
        return free, obstacle, cliff, agent, goal

    def print(self):
        rows = []
        for row in range(self.x.shape[0]):
            str = np.array(self.x[row], dtype=np.str)
            rows.append(' '.join(str))
        print('\n'.join(rows))



class Env(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.maze = Maze(**kwargs)
        self.motions = VonNeumannMotion()
        self.size = kwargs['size']
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape= self.maze.size , dtype=np.uint8)
        # taking just x and y coordinates as observation space
        #self.observation_space = Box(low=0, high=self.maze.size[0], shape=(2,), dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.start_idx = [[1, 1]]
        self.goal_idx = [[self.size - 2, self.size - 2]]
        
    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        cliff = self._is_cliff(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]
        
        if self._is_goal(new_position):
            reward = 100
            done = True
        elif cliff:
            reward = -100
            done = True
        elif not valid:
            reward = 0
            done = False
        else:
            reward = 0
            done = False
        return self.maze.to_value(), reward, done, {}
        
    def reset(self):
        self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.goal.positions = self.goal_idx
        return self.maze.to_value()
    
    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]] if within_edge else False
        return nonnegative and within_edge and passable
    
    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def _is_cliff(self, position):
        out = False
        for pos in self.maze.objects.cliff.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    
    def get_image(self):
        return self.maze.to_rgb()
        
class Maze_XY(gym.Env):
    def __init__(self, **kwargs):
        self.count = 0
        seed = kwargs['seed']
        np.random.seed(kwargs['seed'])
        self.x = random_sutton_maze(size = kwargs['size'], difficulty = kwargs['difficulty'], seed = kwargs['seed'])
        while not check_solvability_maze(self.x):
            seed = np.random.randint(0, 10000)
            self.count = self.count + 1
            self.x = random_sutton_maze(size = kwargs['size'], difficulty = kwargs['difficulty'], seed = seed)
            if self.count > 1000:
                raise Exception('No solvable maze found')

        #self.goal = np.array([4, 11])
        self.goal = np.array([self.x.shape[0] - 2, self.x.shape[1] - 2])
        #self.cliff = np.array([[4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10]])
        self.cliff = np.stack(np.where(self.x == 2), axis=1)
        #self.start = np.array([4, 0])
        self.start = np.array([1, 1])
        self._agent_location = self.start
        print("Maze created after {} tries with {} seed".format(self.count, seed))
        print("Goal: {}".format(self.goal))
        print("Cliff: {}".format(self.cliff))
        print("Start: {}".format(self.start))
        print("Maze: {}".format(self.x))    

        self.window_size = 512
        self.window = None
        self.clock = None

        #self.observation_space = Box(low=0, high=self.size[1], shape=(2, ), dtype=np.uint8)
        self.observation_space = Box(low=0, high=self.x.shape[0], shape=(2, ), dtype=np.uint8)
        self.action_space = Discrete(4)
        self._action_to_direction = {
        0: np.array([0, 1]), #right
        1: np.array([1, 0]), #down
        2: np.array([0, -1]), #left
        3: np.array([-1, 0]) #up
        }
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode.
        """
                
    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {}

    def reset(self):
        self._agent_location = self.start
        #if self.render_mode == "human":
        #    self._render_frame()

        return np.array(self._agent_location, dtype=np.uint8)

    def step(self, action):
        new_position = self._agent_location + self._action_to_direction[int(action)]
        is_valid = self._is_valid(new_position)
        is_cliff = self._is_cliff(new_position)
        if is_valid:
            self._agent_location = new_position
        if self._is_goal(new_position):
            reward = 0
            done = True
        elif is_cliff:
            reward = -100
            done = True
        elif not is_valid:
            reward = -1
            done = False
        else:
            reward = -1
            done = False
        #if self.render_mode == "human":
        #    self._render_frame()
        return self._get_obs(), reward, done, self._get_info()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        #within_edge = position[0] < self.size[0] and position[1] < self.size[1]
        within_edge = position[0] < self.x.shape[0] and position[1] < self.x.shape[1]
        passable = not self._is_cliff(position) if within_edge else False
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        if position[0] == self.goal[0] and position[1] == self.goal[1]:
            out = True
        return out

    def _is_cliff(self, position):
        out = False
        for pos in self.cliff:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

class Maze_XY_Img(Maze_XY):
    def __init__(self, **kwargs):
        self.count = 0
        seed = kwargs['seed']
        np.random.seed(kwargs['seed'])
        self.x = random_sutton_maze2(size = kwargs['size'], difficulty = kwargs['difficulty'], seed = kwargs['seed'])
        while not check_solvability_maze(self.x):
            seed = np.random.randint(0, 10000)
            self.count = self.count + 1
            self.x = random_sutton_maze2(size = kwargs['size'], difficulty = kwargs['difficulty'], seed = seed)
            if self.count > 1000:
                raise Exception('No solvable maze found')

        #self.goal = np.array([4, 11])
        self.goal = np.array([self.x.shape[0] - 2, self.x.shape[1] - 2])
        #self.cliff = np.array([[4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10]])
        self.cliff = np.stack(np.where(self.x == 2), axis=1)
        #self.start = np.array([4, 0])
        self.start = np.array([1, 1])
        self._agent_location = self.start
        print("Maze created after {} tries with {} seed".format(self.count, seed))
        print("Goal: {}".format(self.goal))
        print("Cliff: {}".format(self.cliff))
        print("Start: {}".format(self.start))
        print("Maze: {}".format(self.x))    

        self.window_size = 512
        self.window = None
        self.clock = None

        self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = Discrete(4)
        self._action_to_direction = {
        0: np.array([0, 1]), #right
        1: np.array([1, 0]), #down
        2: np.array([0, -1]), #left
        3: np.array([-1, 0]) #up
        }
        self.size = np.array([kwargs['size'], kwargs['size']])


    def step(self, action):
        new_position = self._agent_location + self._action_to_direction[int(action)]
        is_valid = self._is_valid(new_position)
        is_cliff = self._is_cliff(new_position)
        if is_valid:
            self._agent_location = new_position
        if self._is_goal(new_position):
            reward = 100
            done = True
        elif is_cliff:
            reward = -100
            done = True
        elif not is_valid:
            reward = 0
            done = False
        else:
            reward = 0
            done = False
        return self._get_obs(), reward, done, self._get_info()

    def _get_obs(self):
        img =  self.render(mode="rgb_array")
        img = cv2.resize(img, (84, 84))
        return img
    
    def reset(self):
        self._agent_location = self.start
        return self._get_obs()

    def render(self, mode="human"):
        self.render_mode = mode
        if  self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame
        if self.window == None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.size[1]*50, self.size[0]*50))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.size[1]*50, self.size[0]*50))
        canvas.fill((255, 255, 255))
        pix_square_size_x = (
            50
        )  # The size of a single grid square in pixels
        pix_square_size_y = (
            50
        )  # The size of a single grid square in pixels
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                (self.goal[1] * 50, self.goal[0] * 50),
                (50, 50),
            ),
        )
        agent_location_swap = np.array([self._agent_location[1], self._agent_location[0]])
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_location_swap + 0.5) * 50,
            50 / 3,
        )
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self._is_cliff(np.array([x, y])):
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            (y * 50, x * 50),
                            (50, 50),
                        ),
                    )
        
        for x in range(self.size[0]):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, x * 50),
                (self.size[1] * 50, x * 50),
            )

        for y in range(self.size[1]):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (y * 50, 0),
                (y * 50, self.size[0] * 50),
            )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
