import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


class MazeGenerator:
    def __init__(self, maze_height, maze_width, animation=False, endpoints=True):        
        # initialize action space
        self.action_space = np.array([[0, 1],
                             [0, -1],
                             [1, 0],
                             [-1, 0]])
        
        # create GIF or not
        self.animation = animation

        # set endpoint color flag
        self.endpoints = endpoints

        # save width and height attributes        
        self.width, self.height = maze_width, maze_height

        # set endpoint position
        self.finish = self.get_finish()

        # default starting position in top left corner
        self.start_position = np.array([0,0])

        # generate maze using DFS from start position
        self.generate_maze(self.start_position)

    def dfs(self, position):
        # randomize step order
        steps = np.random.permutation(self.action_space)

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
                        self.intermediates.append(np.array(self.maze))

                    # continue DFS
                    self.dfs(next_position)

    def generate_maze(self, position):
        # save intermediate mazes for animation
        self.intermediates = []

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
    
    def animate(self):
        if not self.animation:
            raise RuntimeError("Animation is not enabled.")
        
        # save all frames as images
        for frame in self.intermediates:
            self.save_frame(frame)

        # make and save gif
        self.create_gif_and_cleanup()

    def save_frame(self, array):
        # filename for frame
        filename = f'frame_{len(self.frames)}.png'

        # setting color bounds
        bounds = [0, 1, 2, 3, 4]
        cmap = mcolors.ListedColormap(['white', 'black', 'red', 'green'])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # plotting settings
        plt.imshow(array, cmap=cmap, norm=norm)
        plt.axis('off')

        # save and close image
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        # append filename
        self.frames.append(filename)
        
    def create_gif_and_cleanup(self, duration=0.5):
        # read in all simages
        images = []
        for filename in self.frames:
            images.append(imageio.imread(filename))
            
        # save into gif
        imageio.mimsave('maze_generation.gif', images, duration=duration)
        
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
    

        

