import sys
import pygame
import numpy as np
import gymnasium as gym

# Step 0: Import images for the environment
# -------
# Paths to your images
agent_img_path = r'Photos\Mario.png'
goal_img_path = r"Photos\Goal.png"
hell_img_path = r'Photos\Bomb.png'
food_img_path = r'Photos\Food.png'

# Step 1: Define your own custom environment
# -------
class PadmEnv(gym.Env):
    def __init__(self, grid_size=6, goal_coordinates=(5,5)) -> None:
        super(PadmEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = None
        self.reward = 0
        self.info = {}
        self.done = False
        self.goal = np.array(goal_coordinates)
        self.hell_states = []
        self.obstacle_states = []
        self.food_states = []
        self.food1 = False
        self.food2 = False
        self.food3 = False

        # Load images
        self.agent_image = pygame.transform.scale(pygame.image.load(agent_img_path), (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(pygame.image.load(goal_img_path), (self.cell_size, self.cell_size))
        self.hell_image = pygame.transform.scale(pygame.image.load(hell_img_path), (self.cell_size, self.cell_size))
        self.food_image = pygame.transform.scale(pygame.image.load(food_img_path), (self.cell_size, self.cell_size))

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size * self.grid_size, self.cell_size * self.grid_size))

    def reset(self):
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0
        self.food1 = False
        self.food2 = False
        self.food3 = False
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.info

    def add_hell_states(self, hell_state_coordinates):
        self.hell_states.append(np.array(hell_state_coordinates))

    def add_food_states(self, food_state_coordinates):
        self.food_states.append(np.array(food_state_coordinates))
    

    def step(self, action):
        self.reward = 0 # Reset reward for each step
        # Agent movement
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] -= 1
        if action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1
        if action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1
        if action == 3 and self.state[1] > 0:  # Left
            self.state[1] -= 1

        # Check goal condition
        if np.array_equal(self.state, self.goal):
            self.reward += 100
            self.done = True
        
        # Check Hell states
        elif any(np.array_equal(self.state, hell) for hell in self.hell_states):
            self.reward += -50
            self.done = True

        # Check food states
        elif any(np.array_equal(self.state, food) for food in self.food_states):

            if np.array_equal(self.state, self.food_states[0]) and not self.food1:
                self.reward += 10
                self.food1 = True
            elif np.array_equal(self.state, self.food_states[0]) and self.food1:
                self.reward += -1

            if np.array_equal(self.state, self.food_states[1]) and not self.food2:
                self.reward += 10
                self.food2 = True
            elif np.array_equal(self.state, self.food_states[1]) and self.food2:
                self.reward += -1

            if np.array_equal(self.state, self.food_states[2]) and not self.food3:
                self.reward += 10
                self.food3 = True
            elif np.array_equal(self.state, self.food_states[2]) and self.food3:
                self.reward += -1

        else:  # Every other state
            self.reward += -0.1

         # Calculate distance to goal
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )
 
        return self.state, self.reward, self.done, self.info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((80,141,78))   # Fill the background
        

        # Draw grid Lines
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), grid, 1)

         # Draw the Goal and agent
        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))
        self.screen.blit(self.agent_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))

        # Draw Hell States
        for hell in self.hell_states:
            self.screen.blit(self.hell_image, (hell[1] * self.cell_size, hell[0] * self.cell_size))

        # Draw Food States
        for food in self.food_states:
            self.screen.blit(self.food_image, (food[1] * self.cell_size, food[0] * self.cell_size))        

        pygame.display.flip()

    def close(self):
        pygame.quit()

def create_env(goal_coordinates, hell_state_coordinates, food_state_coordinates):

    env = PadmEnv(goal_coordinates=goal_coordinates)

    for i in range(len(hell_state_coordinates)):
        env.add_hell_states(hell_state_coordinates[i])

    for i in range(len(food_state_coordinates)):
        env.add_food_states(food_state_coordinates[i])
        
    return env 