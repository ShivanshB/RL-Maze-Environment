import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

class MazeGenerator:
    def __init__(self, maze_height, maze_width, animation=False):        
        # initialize action space
        self.action_space = np.array([[0, 1],
                             [0, -1],
                             [1, 0],
                             [-1, 0]])
        
        # create GIF or not
        self.animation = animation

        # save width and height attributes        
        self.width, self.height = maze_width, maze_height

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

        # plotting settings
        plt.imshow(array, cmap='inferno_r')
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

        print(self.intermediates[-1])

        # save into gif
        imageio.mimsave('maze_generation.gif', images, duration=duration)
        
        # delete the frames
        for filename in self.frames:
            os.remove(filename)

        # clear the frames list
        self.frames = []
    

        

