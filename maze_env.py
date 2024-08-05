import os
import gym
from gym import spaces
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


class MazeEnv(gym.Env):
    def __init__(self, maze_height, maze_width, animation=False, endpoints=True, max_frames=3000):        
        # initialize directions
        self.directions = np.array([[0, 1],
                             [0, -1],
                             [1, 0],
                             [-1, 0]])
        
        # setting color bounds
        bounds = [0, 1, 2, 3, 4, 5]
        self.cmap = mcolors.ListedColormap(['white', 'black', 'red', 'green', 'orange'])
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)
        
        # create animated GIFs or not
        self.animation = animation

        # set max_frames
        self.max_frames = max_frames

        # initialize animation storage
        self.intermediates = {'generation': [],
                              'training': []}

        # set endpoint color flag
        self.endpoints = endpoints

        # save width and height attributes        
        self.width, self.height = maze_width, maze_height

        # calculate endpoint position
        self.finish = self.get_finish()

        # default starting position in top left corner
        self.start_position = np.array([0,0])

        # start agent in starting position
        self.agent_position = self.start_position

        # four possible actions: left right up down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(maze_height, maze_width), shape=(2,), dtype=np.uint8)

        # generate maze using DFS from start position
        self.generate_maze(self.start_position)


    def reset(self , seed=None, options=None):
        # reset start, end, and current agent position
        self.maze[tuple(self.agent_position)] = 0
        self.maze[tuple(self.start_position)] = 3
        self.maze[tuple(self.finish)] = 2

        # reset agent position to beginning
        self.agent_position = self.start_position

        # set starting position to agent value for visualizations
        self.maze[tuple(self.start_position)] = 4

        # reset animations per episode
        self.intermediates['training'] = [np.array(self.maze)]
        
        observation = tuple(self.agent_position)
        info = {}

        return observation, info
    
    def is_valid_position(self, position):
        if (0 <= position[0] < self.height and 0 <= position[1] < self.width and self.maze[tuple(position)] != 1):
            return True
        return False
    
    def step(self, action):
        # execute the action
        new_position = self.agent_position + self.directions[action]

        if self.is_valid_position(new_position):
            # clear previous position     
            self.maze[tuple(self.agent_position)] = 0 

            # set start and end squares
            self.maze[tuple(self.start_position)] = 3
            self.maze[tuple(self.finish)] = 2

            # move agent to new position 
            self.agent_position = new_position
            self.maze[tuple(self.agent_position)] = 4

            # add to animations
            self.intermediates['training'].append(np.array(self.maze))

        # check if agent is finished
        done = np.all(self.agent_position == self.finish)

        # calculate reward
        if done:
            reward = 10.00
        # penalize trying to go through walls
        elif self.is_valid_position(new_position) and self.maze[tuple(new_position)] == 1:
            reward = -1.00
        else:
            reward = -0.10

        observation = tuple(self.agent_position)
        info = {}

        return observation, reward, done, False, info
    
    def render(self):
        self.save_frame(self.maze, mode='render')

    def dfs(self, position):
        # randomize step order
        steps = np.random.permutation(self.directions)

        # take each step, using DFS
        for step in steps:
            # step over twice to avoid grid walls
            next_position = position + (2 * step)
            if all(next_position < self.maze.shape) and all(0 <= next_position):
                if tuple(next_position) not in self.visited:
                    # set the square in between to be empty
                    self.maze[tuple(next_position - step)] = 0

                    # if endpoints, set ending position to red
                    if self.endpoints and all(self.finish == next_position):
                        self.maze[tuple(next_position)] = 2

                    # add next position to visited
                    self.visited.add(tuple(next_position))
                    
                    # if animate, add copy to intermediate maze array for later animation
                    if self.animation:
                        self.intermediates['generation'].append(np.array(self.maze))

                    # continue DFS
                    self.dfs(next_position)

    def generate_maze(self, position):
        # save intermediate mazes for animation
        self.intermediates['generation'] = []

        # save frame filenames
        self.frames = []

        # initialize maze with starting state of grid of walls
        self.maze = np.ones((self.height, self.width))
        self.maze[::2, ::2] = 0

        # save visited locations for DFS
        self.visited = set(tuple(self.start_position))

        # if endpoints, set initial value to green
        if self.endpoints:
            self.maze[0, 0] = 3

        # run dfs from position
        self.dfs(position)
    
    def animate(self, mode='generation'):
        # reset frames for animating
        self.frames = []

        if not self.animation:
            raise RuntimeError("Animation is not enabled.")
        
        # save all frames as images
        for i, frame in enumerate(self.intermediates[mode]):
            # break and save if animation too long
            if i > self.max_frames:
                break
            self.save_frame(frame)

        # make and save gif
        self.create_gif_and_cleanup(mode=mode)

    def save_frame(self, array, mode='animate'):
        # filename for frame
        if mode == 'animate':
            filename = f'frame_{len(self.frames)}.png'
        elif mode == 'render':
            filename = 'render.png'

        # plotting settings
        plt.imshow(array, cmap=self.cmap, norm=self.norm)
        plt.axis('off')

        # save and close image
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        # append filename
        self.frames.append(filename)
        
    def create_gif_and_cleanup(self, duration=0.5, mode='generation'):
        # read in all simages
        images = []
        for filename in self.frames:
            images.append(imageio.imread(filename))
            
        # save into gif
        imageio.mimsave(f'{mode}.gif', images, duration=duration)
        
        # delete the frames
        for filename in self.frames:
            os.remove(filename)

        # clear the frames list
        self.frames = []

    def get_finish(self):
        # cleaner notation
        r, c = self.height, self.width

        if r % 2 == 1 and c % 2 == 1:
            return np.array([r - 1, c - 1])
        elif r % 2 == 0 and c % 2 == 0:
            return np.array([r - 2, c - 2])
        elif r % 2 == 1 and c % 2 == 0:
            return np.array([r - 1, c - 2])
        else:
            return np.array([r - 2, c - 1])
    

        

